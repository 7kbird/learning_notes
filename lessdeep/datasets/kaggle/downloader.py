import os
import pickle
import re

import progressbar
from mechanicalsoup import Browser
from ...datasets import default_dir
from ...datasets import base


def login(username, password, browser=None):
    pickle_path = os.path.join(default_dir('kaggle', 'cache'),
                               'browser.pickle')

    # if not browser and os.path.isfile(pickle_path):
    #     with open(pickle_path, 'rb') as file:
    #         data = pickle.load(file)
    #         if data['username'] == username and \
    #                 data['password'] == password:
    #             return data['browser']
    # else:
    #     os.makedirs(os.path.dirname(pickle_path), exist_ok=True)

    login_url = 'https://www.kaggle.com/account/login'
    if not browser:
        browser = Browser()

    login_page = browser.get(login_url)

    token = re.search(
        'antiForgeryToken: \'(?P<token>.+)\'',
        str(login_page.soup)
    ).group(1)

    login_result_page = browser.post(
        login_url,
        data={
            'username': username,
            'password': password,
            '__RequestVerificationToken': token
        }
    )

    error_match = re.search(
        '"errors":\["(?P<error>.+)"\]',
        str(login_result_page.soup)
    )

    if error_match:
        error = error_match.group(1)
        raise RuntimeError('There was an error logging in: ' + error)

    # with open(pickle_path, 'wb') as f:
    #     pickle.dump(dict(
    #         username=username, password=password, browser=browser
    #     ), f)

    return browser


def ui_login(browser=None):
    import getpass

    user = getpass.getpass("Enter your user name:")
    password = getpass.getpass("Enter your Password:")

    return login(user, password, browser)


def download_dataset(competition, filename, folder='.', browser=None):
    if not browser:
        browser = Browser()

    # TODO: add more
    base = 'https://www.kaggle.com'
    data_url = '/'.join([base, 'c', competition, 'data'])

    data_page = browser.get(data_url)

    data = str(data_page.soup)
    links = re.findall(
        '"url":"(/c/{}/download/[^"]+)"'.format(competition), data
    )

    if not links:  # fallback for inclass competition
        links = map(
            lambda link: link.get('href'),
            data_page.soup.find(id='data-files').find_all('a')
        )

    if not links:
        print('not found')

    for link in links:
        url = base + link
        if url.endswith('/' + filename):
            return download_file(browser, url, folder)
    raise RuntimeError("Cannot found {0} for competition {1}".format(filename, competition))


def _is_downloadable(response):
    '''
    Checks whether the response object is a html page
    or a likely downloadable file.
    Intended to detect error pages or prompts
    such as kaggle's competition rules acceptance prompt.
    Returns True if the response is a html page. False otherwise.
    '''

    content_type = response.headers.get('Content-Type', '')
    content_disp = response.headers.get('Content-Disposition', '')

    if 'text/html' in content_type and 'attachment' not in content_disp:
        # This response is a html file
        # which is not marked as an attachment,
        # so we likely hit a rules acceptance prompt
        return False
    return True


def download_file(browser, url, download_folder='.', try_login=True):
    local_filename = url.split('/')[-1]
    final_file = os.path.join(download_folder, local_filename)
    down_file = final_file + '.download'

    if os.path.isfile(final_file):
        print('{} already downloaded !'.format(final_file))
        return final_file

    # Make path first so one can manually copy file from other place when
    # network is slow
    os.makedirs(download_folder, exist_ok=True)

    print('downloading {}\n'.format(url))
    retry = 3
    while retry > 0:
        retry -= 1
        header_req = browser.request('head', url)
        if '/account/login?' in header_req.url:
            if try_login:
                ui_login(browser)
            else:
                raise RuntimeError("Not login")
    content_length = int(header_req.headers.get('Content-Length'))

    widgets = [local_filename, ' ', progressbar.Percentage(), ' ',
               progressbar.Bar(marker='#'), ' ',
               progressbar.ETA(), ' ', progressbar.FileTransferSpeed()]

    headers = {}
    file_size = 0
    if os.path.isfile(down_file):
        file_size = os.path.getsize(down_file)
        if file_size < content_length:
            headers['Range'] = 'bytes={}-'.format(file_size)

    finished_bytes = file_size

    if file_size == content_length:
        os.rename(down_file, final_file)
        print('{} already downloaded !'.format(final_file))
        return final_file
    elif file_size > content_length:
        raise RuntimeError('Something wrong here, Incorrect file !')
    else:
        bar = progressbar.ProgressBar(widgets=widgets,
                                      maxval=content_length).start()
        bar.update(finished_bytes)

    stream = browser.get(url, stream=True, headers=headers)
    if not _is_downloadable(stream):
        warning = (
            'Warning:'
            'download url for file {} resolves to an html document'
            'rather than a downloadable file. \n'
            'See the downloaded file for details.'
            'Is it possible you have not'
            'accepted the competition\'s rules on the kaggle website?'
                .format(local_filename)
        )
        print('{}\n'.format(warning))

    os.makedirs(os.path.dirname(down_file), exist_ok=True)
    with open(down_file, 'ab') as f:
        for chunk in stream.iter_content(chunk_size=1024):
            if chunk:  # filter out keep-alive new chunks
                f.write(chunk)
                finished_bytes += len(chunk)
                bar.update(finished_bytes)
    os.rename(down_file, final_file)
    bar.finish()

    return final_file


def download_extract(competition, filename, extract_root, download_dir='.',
                     browser=None):
    down_file = download_dataset(competition, filename, download_dir, browser)

    extract_dir = os.path.join(extract_root, filename.split('.')[0])
    if not os.path.exists(extract_dir):
        base.extract(down_file, extract_dir)
    else:
        print('Already extracted: ' + filename)
