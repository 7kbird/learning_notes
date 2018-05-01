import os
import zipfile
import tarfile
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import tempfile


class MultiprocessExtractor(object):
    def extract_members(self, members, file_path, dst_root):
        """extract member and return total number"""
        raise NotImplementedError()

    def list_members(self, file_path):
        raise NotImplementedError()


class ZipExtractor(MultiprocessExtractor):
    def extract_members(self, members, file_path, dst_root):
        with open(file_path, 'rb') as f:
            zf = zipfile.ZipFile(f)
            for mem in members:
                zf.extract(mem, dst_root)
            return len(members)

    def list_members(self, file_path):
        with open(file_path, 'rb') as f:
            return zipfile.ZipFile(f).infolist()


class TarExtractor(MultiprocessExtractor):
    def extract_members(self, members, file_path, dst_root):
        tf = tarfile.TarFile(file_path)
        for mem in members:
            tf.extract(mem, dst_root)
        return len(members)

    def list_members(self, file_path):
        tf = tarfile.TarFile(file_path)
        return tf.getmembers()


def _chunks(l, step):
    chunk = []
    for item in l:
        if len(chunk) < step:
            chunk.append(item)
        if len(chunk) == step:
            yield chunk
            chunk = []
    if chunk:
        yield chunk


def extract_file(file_path, extract_dir, remove_one=True, progressbar=True,
                 extractor=None):

    """
    Please check if ``extract_dir`` already exist. This extract will
    merege all extract file into target folder if already exist
    """

    # Prepare extractor
    if extractor:
        if isinstance(extractor, MultiprocessExtractor):
            raise ValueError('extractor must be a subclass of MultiprocessExtractor')
    else:
        if zipfile.is_zipfile(file_path):
            extractor = ZipExtractor()
        elif tarfile.is_tarfile(file_path):
            extractor = TarExtractor()
        else:
            raise ValueError('File is not supported: ' + file_path)

    # Create temporary directory aside of the ``extract_dir``
    ex_dir, ex_base = os.path.split(extract_dir)
    if not ex_base:
        ex_dir, ex_base = os.path.split(ex_dir)
    extract_tmp = tempfile.mkdtemp(prefix=ex_base + '_extract', dir=ex_dir)

    # Members split into chunks and process in multi-process
    all_members = list(extractor.list_members(file_path))
    step = len(all_members) / multiprocessing.cpu_count()
    step = max(2, step // 20)

    with ProcessPoolExecutor() as pool:
        futures = [pool.submit(extractor.extract_members, members, file_path, extract_tmp)
                   for members in _chunks(all_members, step)]
        if progressbar:
            from yama.util import tqdm
            bar = tqdm(total=len(all_members), unit='',
                       postfix={'extract': ex_base})
        for future in as_completed(futures):
            complete_num = future.result()
            if progressbar:
                bar.update(complete_num)
