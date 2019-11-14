import base64
import xmlrpc.client as xmlrpclib
import time
import re
import numpy as np
from scipy.io import savemat


neos = xmlrpclib.ServerProxy('https://neos-server.org:3333')


def neos_sdpt3_solve(input_file, output_file):
    """
    Sends sdpt3 job to NEOS and returns results.

    :param input_file:  Path to mat file with sedumi solver format.
    :return:            Results of NEOS job.
    """

    # Generate xml for job.
    xml = generate_sdpt3_xml(input_file)

    # Send the job to NEOS.
    job_number, password = send_neos_job(xml)

    # Get results.
    results = get_job_results(job_number, password)

    # Get the optimizer from results.
    optimizer = extract_sdpt3_optimizer(results)

    # Save results in output_file and return.
    savemat(output_file, {"optimizer": optimizer})

    return optimizer


def extract_sdpt3_optimizer(job_results):
    """
    NEOS results are returned in large file with other, irrelevant information. This
    function extracts the optimizer (the data of interest) and returns it as numpy
    vector.

    :param job_results: Results from NEOS XML server.
    :return:            Vector containing value of optimizer from job.
    """

    # Get data string from result.
    data = str(job_results.data)

    # Get string containing optimizer "y".
    optimizer_string = data[data.rfind('y'):]

    # Strip everything except data values.
    optimizer_string = re.sub('[\\\\ny=]', '', optimizer_string)
    optimizer_string = optimizer_string[:optimizer_string.find('>>')]

    # Turn into array of floats.
    optimizer_array = [float(x) for x in optimizer_string.split()]

    # Turn into numpy array and return it.
    return np.array(optimizer_array).reshape(-1, 1)


def get_job_results(job_number, password):
    """
    Waits for a submitted job to be completed and returns results when completed.

    :param job_number: Number of job.
    :param password:   Password for job.
    :return:           Job results.
    """

    # Poll status, keep sleeping for 1 second until completed.
    while neos.getJobStatus(job_number, password) != "Done":
        time.sleep(1)

    return neos.getFinalResults(job_number, password)


def send_neos_job(xml):
    """
    Submits a NEOS job determined by xml text.

    :param xml: XML job to send to NEOS server.
    :return:    (job_number, password)
    """

    return neos.submitJob(xml)


def generate_sdpt3_xml(sedumi_mat_file):
    """
    Generates xml for sdpt3 NEOS job.

    :param sedumi_mat_file: Path to sedumi format file.
    :return:                XML for sdpt3 job on NEOS.
    """

    # Read sedumi format contents (this is placed into XML file).
    test_mat = open(sedumi_mat_file, 'rb')
    mat_data = str(base64.b64encode(test_mat.read()))[2:][:-1]
    test_mat.close()

    # Build xml string.
    xml = ('<document>'
           '<category>sdp</category>'
           '<solver>sdpt3</solver>'
           '<inputType>MATLAB_BINARY</inputType>'
           '<client>Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/78.0.3904.70 Safari/537.36@128.61.84.87</client>'
           '<priority>long</priority>'
           '<email></email>'
           '<dat><![CDATA[]]></dat>\n\n'
           '<mat><base64>\n')
    xml += mat_data
    xml += ('</base64></mat>\n\n'
            '<DTYPE><![CDATA[hkm]]></DTYPE> \n\n'
            '<comment><![CDATA[]]></comment>\n\n'
            '</document>')

    return xml


if __name__ == "__main__":

    results = neos_sdpt3_solve('test.mat')

    print(results)
