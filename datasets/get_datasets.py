import zipfile
import urllib.request
from tqdm import tqdm

class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)

url = "http://liutkus.net/DSD100.zip"
# url = "https://www.loria.fr/~aliutkus/DSD100subset.zip"

output_path = 'DSD100.zip'

with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, 
                         desc=url.split('/')[-1]) as t:
    urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)

with zipfile.ZipFile('DSD100.zip', 'r') as f:
    f.extractall()
