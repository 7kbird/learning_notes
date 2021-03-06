import unittest
from unittest import TestCase
import tempfile
import os
import warnings
import shutil
from yama.util.data import download_url, extract_file
from threading import Thread
from http.server import SimpleHTTPRequestHandler
from rangeserver import RangeRequestHandler, ThreadingHTTPServer
import threading
import hashlib


class FileServerThread(Thread):
    def __init__(self, port, root, handler):
        self.root = root
        self.port = port
        self.server = None
        self.handler = handler
        super().__init__()

    def run(self):
        os.chdir(self.root)
        ThreadingHTTPServer.allow_reuse_address = True

        httpd = ThreadingHTTPServer(("127.0.0.1", self.port), self.handler)
        try:
            self.server = httpd
            #print("serving at port", self.port)
            httpd.serve_forever()
        finally:
            httpd.shutdown()
            httpd.server_close()


class TestDownloadUrl(TestCase):
    src_dir = ''
    src_file = ''
    url = ''
    file_md5 = ''
    port = 8887

    @classmethod
    def setUpClass(cls):
        cls.src_dir = tempfile.mkdtemp()
        file = 'moc_src.tar.gz'
        cls.url = 'http://127.0.0.1:{}/{}'.format(cls.port, file)

        md5 = hashlib.md5()
        file_size = 200 * 1024 * 1024
        cls.src_file = os.path.join(cls.src_dir, file)
        with open(os.path.join(cls.src_dir, file), 'wb') as f:
            block_size = 1024
            for _ in range(file_size // block_size):
                buff = os.urandom(block_size)
                f.write(buff)
                md5.update(buff)
        cls.file_md5 = md5.hexdigest()

    @classmethod
    def tearDownClass(cls):
        if cls.src_dir:
            shutil.rmtree(cls.src_dir, ignore_errors=True)

    def setUp(self):
        self.down_dir = tempfile.mkdtemp()

    def tearDown(self):
        if self.server_thread and self.server_thread.server:
            self.server_thread.server.shutdown()
            self.server_thread.join()
        shutil.rmtree(self.down_dir, ignore_errors=True)

    def test_download_multi_thread(self):
        req_handle = RangeRequestHandler
        req_handle.log_enabled = False
        self.server_thread = FileServerThread(self.port, self.src_dir, req_handle)
        self.server_thread.start()
        file, down = download_url(self.url, self.down_dir, 'test_down.tar.gz',
                                  md5=self.file_md5, threads=2)
        self.assertEqual(file, os.path.join(self.down_dir, 'test_down.tar.gz'))
        self.assertTrue(down)
        self.assertTrue(os.path.exists(file))
        self.assertEqual(len(os.listdir(self.down_dir)), 1)

    def test_download_multi_thread_resume(self):
        req_handle = RangeRequestHandler
        req_handle.log_enabled = False
        self.server_thread = FileServerThread(self.port, self.src_dir, req_handle)
        self.server_thread.start()

        # Stop server after downloading only some parts
        def shutdown_server():
            self.server_thread.server.shutdown()
        threading.Timer(1, shutdown_server).start()

        import urllib3
        with self.assertRaises(urllib3.exceptions.HTTPError) as cm:
            download_url(self.url, self.down_dir, 'test_down.tar.gz',
                         md5=self.file_md5, threads=2)
        self.assertGreater(len(os.listdir(self.down_dir)), 1)

        self.server_thread.join()
        self.server_thread = FileServerThread(self.port, self.src_dir, req_handle)
        self.server_thread.start()
        file, down = download_url(self.url, self.down_dir, 'test_down.tar.gz',
                                  md5=self.file_md5)
        self.assertTrue(os.path.exists(file))
        # TODO: check download size is less than total


class TestExtract(unittest.TestCase):
    src_dir = ''

    @classmethod
    def setUpClass(cls):
        cls.src_dir = tempfile.mkdtemp()
        files = ['root{}'.format(i) for i in range(100)] + \
                ['root_dir{}/ch{}'.format(i, j) for j in range(10) for i in range(10)]
        buff = os.urandom(1024*1024)
        for name in files:
            os.makedirs(os.path.dirname(os.path.join(cls.src_dir, name)), exist_ok=True)
            with open(os.path.join(cls.src_dir, name), 'wb') as f:
                f.write(buff)

    @classmethod
    def tearDownClass(cls):
        if cls.src_dir:
            shutil.rmtree(cls.src_dir, ignore_errors=True)

    def setUp(self):
        self.work_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.work_dir, ignore_errors=True)

    def test_extract_zip(self):
        zip_path = os.path.join(self.work_dir, 'archive')
        shutil.make_archive(zip_path, 'zip', self.src_dir)
        zip_path += '.zip'
        extract_dir = os.path.join(self.work_dir, 'ext')
        extract_file(zip_path, extract_dir)

        from filecmp import dircmp
        diff = dircmp(extract_dir, self.src_dir)
        self.assertEqual(0, len(diff.diff_files))

    def test_remove_one(self):
        one_root = os.path.join(self.work_dir, 'one')
        os.mkdir(one_root)
        shutil.copytree(self.src_dir, os.path.join(one_root, 'sub'))
        self.assertEqual(1, len(os.listdir(one_root)))
        self.assertGreater(len(os.listdir(os.path.join(one_root, 'sub'))), 1)

        zip_path = os.path.join(self.work_dir, 'archive')
        shutil.make_archive(zip_path, 'zip', one_root)
        zip_path += '.zip'

        extract_dir = os.path.join(self.work_dir, 'ext1')
        extract_file(zip_path, extract_dir, remove_one=True)

        from filecmp import dircmp
        diff = dircmp(extract_dir, self.src_dir)
        self.assertEqual(0, len(diff.diff_files))


if __name__ == '__main__':
    unittest.main()
