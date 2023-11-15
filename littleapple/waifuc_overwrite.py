import os

from imgutils.tagging import tags_to_text

from waifuc.export.base import LocalDirectoryExporter
from waifuc.model import ImageItem

from waifuc.export import TextualInversionExporter
from waifuc.source.web import WebDataSource
from shutil import rmtree
from typing import Iterator, Tuple, Union
from PIL import UnidentifiedImageError, Image
from PIL.Image import DecompressionBombError
from hbutils.system import urlsplit, TemporaryDirectory
import warnings
from waifuc.utils import download_file
class AppleWebDataSource(WebDataSource):
    def _iter(self) -> Iterator[ImageItem]:
        for id_, url, meta in self._iter_data():
            with TemporaryDirectory() as td:
                _, ext_name = os.path.splitext(urlsplit(url).filename)
                filename = f'{self.group_name}_{id_}{ext_name}'
                td_file = os.path.join(td, filename)
                try:
                    download_file(
                        url, td_file, desc=filename,
                        session=self.session, silent=self.download_silent
                    )
                    image = Image.open(td_file)
                    image.load()
                except UnidentifiedImageError:
                    warnings.warn(f'{self.group_name.capitalize()} resource {id_} unidentified as image, skipped.')
                    continue
                except (IOError, DecompressionBombError) as err:
                    warnings.warn(f'Skipped due to error: {err!r}')
                    continue
                except NotADirectoryError:
                    os.remove(td)
                    continue

                meta = {**meta, 'url': url}
                yield ImageItem(image, meta)






class AppleTextualInversionExporter(TextualInversionExporter):
    def __init__(self, output_dir: str, clear: bool = False,
                 use_spaces: bool = False, use_escape: bool = True,
                 include_score: bool = False, score_descend: bool = True,
                 skip_when_image_exist: bool = False):
        super().__init__(output_dir, clear, use_spaces, use_escape, include_score, score_descend, skip_when_image_exist)

    def export_item(self, item: ImageItem):
        if 'filename' in item.meta:
            filename = item.meta['filename']
        else:
            self.untitles += 1
            filename = f'untited_{self.untitles}.png'

        tags = item.meta.get('tags', None) or {}

        full_filename = os.path.join(self.output_dir, filename)
        full_tagname = os.path.join(self.output_dir, os.path.splitext(filename)[0] + '.txt')
        full_directory = os.path.dirname(full_filename)
        if full_directory:
            os.makedirs(full_directory, exist_ok=True)

        if not self.skip_when_image_exist or not os.path.exists(full_filename) or not os.path.splitext(filename)[1] == '.gif':
            item.image.save(full_filename)
        with open(full_tagname, 'w', encoding='utf-8') as f:
            f.write(tags_to_text(tags, self.use_spaces, self.use_escape, self.include_score, self.score_descend))