import logging
import os
from typing import Iterator, Optional, Union, Tuple, Literal

from hbutils.system import urlsplit
from pixivpy3 import AppPixivAPI

from .web import WebDataSource
from ..utils import get_requests_session

_FILTER = Literal["for_ios", ""]
_TYPE = Literal["illust", "manga", ""]
_RESTRICT = Literal["public", "private", ""]
_CONTENT_TYPE = Literal["illust", "manga", ""]
_MODE = Literal[
    "day",
    "week",
    "month",
    "day_male",
    "day_female",
    "week_original",
    "week_rookie",
    "day_manga",
    "day_r18",
    "day_male_r18",
    "day_female_r18",
    "week_r18",
    "week_r18g",
    "",
]
_SEARCH_TARGET = Literal[
    "partial_match_for_tags", "exact_match_for_tags", "title_and_caption", "keyword", ""
]
_SORT = Literal["date_desc", "date_asc", "popular_desc", ""]
_DURATION = Literal[
    "within_last_day", "within_last_week", "within_last_month", "", None
]
_BOOL = Literal["true", "false"]
_SELECT = Literal['square_medium', 'medium', 'large', 'original']


class BasePixivSource(WebDataSource):
    def __init__(self, group_name: str = 'pixiv', select: _SELECT = 'large',
                 no_ai: bool = False, refresh_token: Optional[str] = None, download_silent: bool = True):
        self.select = select
        self.no_ai = no_ai
        self.refresh_token = refresh_token
        self.client = AppPixivAPI()
        self.client.requests = get_requests_session(session=self.client.requests)
        self.client.requests.headers.update({"Referer": "https://app-api.pixiv.net/"})
        WebDataSource.__init__(self, group_name, self.client.requests, download_silent)

    def _iter_illustration(self) -> Iterator[dict]:
        raise NotImplementedError  # pragma: no cover

    def _iter_data(self) -> Iterator[Tuple[Union[str, int], str, dict]]:
        if self.refresh_token:
            self.client.auth(refresh_token=self.refresh_token)

        for illust in self._iter_illustration():
            if illust['type'] != 'illust':
                continue
            if self.no_ai and illust['illust_ai_type'] == 2:
                continue

            if illust['page_count'] == 1:
                if self.select != 'original':
                    urls = [illust['image_urls'][self.select]]
                else:
                    urls = [illust['meta_single_page']['original_image_url']]

            else:
                urls = [page['image_urls'][self.select] for page in illust['meta_pages']]

            for i, url in enumerate(urls):
                _, ext_name = os.path.splitext(urlsplit(url).filename)
                filename = f'{self.group_name}_{illust["id"]}_{i}{ext_name}'
                meta = {
                    'pixiv': illust,
                    'group_id': f'{self.group_name}_{illust["id"]}',
                    'instance_id': f'{self.group_name}_{illust["id"]}_{i}',
                    'filename': filename,
                }
                yield f'{illust["id"]}_{i}', url, meta


class PixivSearchSource(BasePixivSource):
    def __init__(self, word: str, search_target: _SEARCH_TARGET = "partial_match_for_tags",
                 sort: _SORT = "date_desc", duration: _DURATION = None, start_date: Optional[str] = None,
                 end_date: Optional[str] = None, filter: _FILTER = "for_ios", req_auth: bool = True,
                 group_name: str = 'pixiv', select: _SELECT = 'large',
                 no_ai: bool = False, refresh_token: Optional[str] = None, download_silent: bool = True):
        BasePixivSource.__init__(self, group_name, select, no_ai, refresh_token, download_silent)
        self.word = word
        self.search_target = search_target
        self.sort = sort
        self.duration = duration
        self.start_date = start_date
        self.end_date = end_date
        self.filter = filter
        self.req_auth = req_auth

    def _iter_illustration(self) -> Iterator[dict]:
        offset = 0
        while True:
            data = self.client.search_illust(self.word, self.search_target, self.sort, self.duration,
                                             self.start_date, self.end_date, self.filter, offset, self.req_auth)
            if 'illusts' not in data:
                logging.warning(f'Illusts not found in page (offset: {offset!r}), skipped: {data!r}.')
                break
            illustrations = data['illusts']
            yield from illustrations

            offset += len(illustrations)
            if not illustrations:
                break


class PixivUserSource(BasePixivSource):
    def __init__(self, user_id: Union[int, str], type: _TYPE = "illust",
                 filter: _FILTER = "for_ios", req_auth: bool = True,
                 group_name: str = 'pixiv', select: _SELECT = 'large',
                 no_ai: bool = False, refresh_token: Optional[str] = None, download_silent: bool = True):
        BasePixivSource.__init__(self, group_name, select, no_ai, refresh_token, download_silent)
        self.user_id = user_id
        self.type = type
        self.filter = filter
        self.req_auth = req_auth

    def _iter_illustration(self) -> Iterator[dict]:
        offset = 0
        while True:
            data = self.client.user_illusts(self.user_id, self.type, self.filter, offset, self.req_auth)
            if 'illusts' not in data:
                logging.warning(f'Illusts not found in page (offset: {offset!r}), skipped: {data!r}.')
                break
            illustrations = data['illusts']
            yield from illustrations

            offset += len(illustrations)
            if not illustrations:
                break


class PixivRankingSource(BasePixivSource):
    def __init__(self, mode: _MODE = "day", filter: _FILTER = "for_ios",
                 date: Optional[str] = None, req_auth: bool = True,
                 group_name: str = 'pixiv', select: _SELECT = 'large',
                 no_ai: bool = False, refresh_token: Optional[str] = None, download_silent: bool = True):
        BasePixivSource.__init__(self, group_name, select, no_ai, refresh_token, download_silent)
        self.mode = mode
        self.filter = filter
        self.date = date
        self.req_auth = req_auth

    def _iter_illustration(self) -> Iterator[dict]:
        offset = 0
        while True:
            data = self.client.illust_ranking(self.mode, self.filter, self.date, offset, self.req_auth)
            if 'illusts' not in data:
                logging.warning(f'Illusts not found in page (offset: {offset!r}), skipped: {data!r}.')
                break
            illustrations = data['illusts']
            yield from illustrations

            offset += len(illustrations)
            if not illustrations:
                break
