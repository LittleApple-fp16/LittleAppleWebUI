import os

from imgutils.tagging import tags_to_text

from waifuc.export.base import LocalDirectoryExporter
from waifuc.model import ImageItem

from waifuc.export import TextualInversionExporter


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