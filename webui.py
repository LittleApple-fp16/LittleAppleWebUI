import io
import logging
import sys
import time
import os
import re
import random
import json
import glob
import argparse
from typing import Literal, cast
try:
    import matplotlib
    import gradio as gr
    import numpy
    import webbrowser
    import asyncio
    import tkinter as tk
    from tkinter import filedialog
    import cv2
    import shutil
    from gradio_client import Client
    from pykakasi import kakasi
    from loguru import logger
    from pypinyin import lazy_pinyin
    from PicImageSearch.model import Ascii2DResponse
    from PicImageSearch import Ascii2D, Network
    from littleapple.refresh_token import get_ref_token
    from littleapple.image_link import get_image_links, download_link
    from littleapple.kemono_dl.main import downloader as kemono_dl
    from littleapple.kemono_dl.args import get_args as kemono_args
    from littleapple.train import run_train_plora
    from littleapple.exceptions import DatasetTypeError
    from pixivpy3 import AppPixivAPI, PixivError
    from tqdm import tqdm
    from tqdm.contrib import tzip
    from waifuc.action import HeadCountAction, AlignMinSizeAction, CCIPAction, ThreeStageSplitAction, ModeConvertAction, ClassFilterAction, PersonSplitAction, TaggingAction, RatingFilterAction, NoMonochromeAction, RandomFilenameAction, FirstNSelectAction, FilterSimilarAction, FileExtAction
    from waifuc.export import SaveExporter, TextualInversionExporter
    from waifuc.source import GelbooruSource, PixivSearchSource, ZerochanSource, LocalSource, GcharAutoSource

    from ditk import logging
    from hbutils.system import TemporaryDirectory
    from cyberharem.dataset import save_recommended_tags
    from cyberharem.publish import find_steps_in_workdir
    from cyberharem.utils import get_hf_fs as cyber_get_hf_fs
    from cyberharem.utils import download_file as cyber_download_file
    from cyberharem.publish.civitai import civitai_publish_from_hf
    from cyberharem.publish.huggingface import deploy_to_huggingface
    from huggingface_hub import hf_hub_url
    from huggingface_hub._login import login as hf_login
    from cyberharem.infer.draw import _DEFAULT_INFER_MODEL
    from kohya.train_network import kohya_train_lora

    from PIL import Image
    from imgutils.data import load_image, load_images, rgb_encode, rgb_decode
    from imgutils.tagging import get_wd14_tags, get_mldanbooru_tags, drop_blacklisted_tags, drop_overlap_tags, tags_to_text
    from imgutils.metrics import ccip_difference, ccip_clustering, lpips_clustering
    from imgutils.operate import censor_areas, squeeze, squeeze_with_transparency
    from imgutils.detect import detect_faces, detect_heads, detection_visualize, detect_person
    from imgutils.segment import segment_rgba_with_isnetis
    from imgutils.ocr import detect_text_with_ocr
    from cyberharem.publish.convert import convert_to_webui_lora
except ModuleNotFoundError as e:
    print(f"[致命错误] - 检测到模块丢失: {e}， 正在尝试安装依赖，请等待安装完成后再次打开")
    import subprocess
    if os.name == 'nt':
        subprocess.run(['dependencies.bat'], check=True)
    elif os.name == 'posix':
        subprocess.run(['dependencies.sh'], check=True)
    else:
        print("[错误] - 未知的操作系统")


def download_images(source_type, character_name, p_min_size, p_background, p_class, p_rating, p_crop_person, p_ccip, p_auto_tagging, num_images, p_ai):
    global output_cache
    actions = []
    rating_map = {0: 'safe', 1: 'r15', 2: 'r18'}
    # ratings_to_filter = set(rating_map.values()) - set([rating_map[i] for i in p_rating if i in rating_map])
    ratings_to_filter = set([rating_map[i] for i in p_rating if i in rating_map])
    gr.Info("开始获取数据集")
    logger.info("\n - 开始获取数据集")
    character_list = character_name.split(',')
    for character in character_list:
        character = character.replace(' ', '_')  # 将空格替换为下划线
        save_path = 'dataset/' + character
        if source_type == 'Gelbooru':
            source_init = GelbooruSource([character, 'solo'])
        elif source_type == 'Pixiv':
            if not cfg.get('pixiv_token', ''):
                gr.Warning("Pixiv未登录")
                return "Pixiv访问令牌未设置"
            source_init = PixivSearchSource(
                character,
                no_ai=p_ai,
                refresh_token=cfg.get('pixiv_token', '')
            )
            # actions.append(CCIPAction())
        elif source_type == 'Zerochan':
            source_init = ZerochanSource([character, 'solo'])
        else:  # 自动
            source_init = GcharAutoSource(character, pixiv_refresh_token=cfg.get('pixiv_token', ''))
            # actions.append(CCIPAction())
        if p_class:
            if 0 in p_class:
                actions.append(NoMonochromeAction())
            if 1 in p_class:
                actions.append(ClassFilterAction(['illustration', 'bangumi']))
            # class_to_filter = set(class_map.values()) - set([class_map[i] for i in p_class if i in class_map])
            # class_to_filter = set([class_map[i] for i in p_class if i in class_map])
        if int(num_images) >= 64 and p_ccip:
            actions.append(CCIPAction())
        if p_crop_person:
            actions.append(PersonSplitAction())
        if p_auto_tagging:
            actions.append(TaggingAction(force=True))
        if p_min_size:
            # logger.debug(int(p_min_size))
            actions.append(AlignMinSizeAction(min_size=int(p_min_size)))
        actions.append(FilterSimilarAction('all'))  # lpips差分过滤
        actions.append(ModeConvertAction('RGB', p_background))
        actions.append(HeadCountAction(1))
        # actions.append(RandomFilenameAction(ext='.png'))
        actions.append(FileExtAction(ext='.png'))  # png格式质量无损
        # logger.debug(cast(list[Literal['safe', 'r15', 'r18']], list(ratings_to_filter)))
        if ratings_to_filter != set(rating_map.values()):
            actions.append(RatingFilterAction(ratings=cast(list[Literal['safe', 'r15', 'r18']], list(ratings_to_filter))))
        actions.append(FirstNSelectAction(int(num_images)))
        if source_type == 'Gelbooru' and not p_auto_tagging:
            source_init.attach(*actions).export(  # 只下载前num_images张图片
                SaveExporter(save_path)  # 将图片保存到指定路径
            )
        else:
            source_init.attach(*actions).export(  # 只下载前num_images张图片
                TextualInversionExporter(save_path)  # 将图片保存到指定路径
            )
        # logger.debug(ratings_to_filter)
    gr.Info("数据集获取已结束")
    output_cache = []
    return "已获取数据集"


def dataset_getImg(dataset_name, rep_name=None):  # 确保每个方法中只调用一次 由于tqdm
    global output_cache
    logger.info(" - 加载数据集图像...")
    if dataset_name.endswith(' (kohya)'):
        dataset_path = f'dataset/_kohya/{dataset_name.replace(" (kohya)", "")}/{rep_name}'
    else:
        dataset_path = f"dataset/{dataset_name}"
    images = []
    img_name = []
    for filename in os.listdir(dataset_path):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            img = Image.open(os.path.join(dataset_path, filename))
            if img is not None:
                images.append(img)
                img_name.append(filename)
    # output_cache = images
    images = load_images(images)
    return images, img_name


def download_illust(i_name, i_source, i_maxsize=None):
    global pyapi
    global cfg
    global output_cache
    gr.Info("开始获取数据集")
    maxsize = round(float(i_maxsize), 1) if i_maxsize else None
    try:
        json_result = pyapi.search_user(i_name)
        illust = json_result.user_previews[0].illusts[0]
        kemono_arg = kemono_args()
        kemono_arg["cookies"] = {k: str(v) for k, v in json.loads(cfg['fanbox_cookie']).items()}
        kemono_arg["links"] = ["https://kemono.su/fanbox/user/" + str(illust['user']['id'])]
        kemono_arg["dirname_pattern"] = f"dataset/{i_name}"
        kemono_arg["max_filesize"] = maxsize
        kemono_arg["retry"] = 3
        if cfg.get('proxie_enabled', False):
            kemono_arg["proxie"] = {
                'http': 'http://' + cfg.get('proxie_ip', None) + ':' + cfg.get('proxie_port', None)
            }
        if 0 in i_source:
            logger.info("[信息] - 画师内容获取需要一段时间")
            links = get_image_links(illust['user']['id'], maxsize)
            for url, name in tzip(links[0], links[1], file=sys.stdout, ascii="░▒█", desc=" - 开始下载"):
                if not os.path.exists(f"dataset/{illust['user']['name']}"):
                    os.makedirs(f"dataset/{illust['user']['name']}")
                download_link(url, f"dataset/{illust['user']['name']}/{name}.png")
        # print(">>> %s, origin url: %s" % (illust.title, illust.image_urls['large']))
        # return "已获取"+illust['user']['name']+"画师数据集"
        if 1 in i_source:
            kemono_dl(kemono_arg)
        gr.Info(i_name+" 数据集获取已结束")
        output_cache = []
        return "下载已结束"
    except Exception as exp:
        gr.Warning("数据集获取失败, 请查看控制台")
        logger.error(f"[错误] - 获取失败\n你必须设置Pixiv访问令牌才能获取Pixiv的内容\n你必须设置Kemono令牌才能获取Fanbox的内容\n你必须输入正确的画师名, 错误信息:{exp}")
        output_cache = []
        return "获取失败\n你必须设置Pixiv访问令牌才能获取Pixiv的内容\n你必须设置Kemono令牌才能获取Fanbox的内容\n你必须输入正确的画师名"


def get_fanbox_cookie():
    webbrowser.open(f"https://kemono.su/account/login")


global gelbooru_fast_settings


# def get_danbooru_fast(tags):
#     batches = tags.split("|")
#     global danbooru_fast_settings
#
#     for i, batch in enumerate(batches):
#         # settings = DanbooruSpider.settings
#         danbooru_fast_settings = {
#             'assistant': "danbooru_crawler",
#             'SEARCH_KEYS': "group_sex+doggystyle",
#             'SPIDER_MODULES': ["danbooru_crawler.spiders"],
#             'NEWSPIDER_MODULE': "danbooru_crawler.spiders",
#             'ROBOTSTXT_OBEY': False,
#             'REQUEST_FINGERPRINTER_IMPLEMENTATION': "2.7",
#             'TWISTED_REACTOR': "twisted.internet.asyncioreactor.AsyncioSelectorReactor",
#             'FEED_EXPORT_ENCODING': "utf-8",
#             'CONCURRENT_REQUESTS': 32,
#             'DOWNLOAD_DELAY': 1,
#             'ITEM_PIPELINES': {'scrapy.pipelines.images.ImagesPipeline': 1},
#             'IMAGES_STORE': '../dataset/test',  # 设置图片存储路径
#         }
#         name = batch.replace(" ", "_")
#         danbooru_fast_settings['IMAGES_STORE'] = f'../dataset/{name}'
#         tag = name.replace(",", "+")
#         danbooru_fast_settings['SEARCH_KEYS'] = tag
#         with open('tmp.bin', 'wb') as tmp:
#             pickle.dump(danbooru_fast_settings, tmp)
#         # execute(["scrapy", "crawl", "danbooru"])
#         # process = multiprocessing.Process(target=CrawlerProcess(danbooru_fast_settings))
#         # process.crawl(DanbooruSpider.DanbooruSpider, tag=tag)
#         # process.start()
#         # reactor.run()
#         DanbooruSpider.__main__()


def has_image(got_list):
    if any(isinstance(item, Image.Image) for item in got_list):
        return True
    else:
        return False


def has_area(got_list):
    if not got_list:
        return False
    if isinstance(got_list, list) and all(
        isinstance(item, list) and len(item) == 1 and
        isinstance(item[0], tuple) and len(item[0]) == 3 and
        isinstance(item[0][0], tuple) and len(item[0][0]) == 4 and
        all(isinstance(n, int) for n in item[0][0]) and
        isinstance(item[0][1], str) and
        isinstance(item[0][2], float)
        for item in got_list
    ):
        return True
    return False


def get_output_status(o_cache):
    if has_image(o_cache):
        return "暂存图像结果"+"["+str(len(o_cache))+"] | "
    elif has_area(o_cache):
        return "暂存区域结果"+"["+str(len(o_cache))+"] | "
    elif not o_cache:
        return "运行结果是空的 | "
    else:
        return "运行结果异常 | "


async def illu_getter(pic):
    global cfg
    global output_cache
    gr.Info("开始获取画师信息")
    if cfg.get('proxie_enabled', False):
        proxies = 'http://'+cfg.get('proxie_ip', None)+':'+cfg.get('proxie_port', None)
    else:
        proxies = None
    async with Network(proxies=proxies) as client:
        ascii2d = Ascii2D(
            client=client
        )
        resp = await ascii2d.search(file=pic)
        selected = None
        for i in resp.raw:
            if i.author_url.startswith("https://www.pixiv.net/users/"):
                selected = i
                break
        if selected is None:
            output_cache = []
            gr.Warning("未找到对应画师")
            return "未找到", ""
        else:
            output_cache = []
            gr.Info("画师 "+selected.author+"的作品 "+selected.title)
            return selected.author + " (" + selected.author_url + ") " + "的作品:" + selected.title, selected.author  # re.search(r'\d+$', selected.author_url).group()


def clustering(dataset_name, thre, rep_name=None):
    global output_cache
    if has_image(output_cache):
        images = output_cache
        gr.Info("差分过滤开始处理 <- 缓存")
    else:
        images = dataset_getImg(dataset_name, rep_name)[0]
        gr.Info("差分过滤开始处理 <- 数据集")
    # print(clusters)
    clustered_imgs = []
    added_clusters = set()  # 创建一个集合 其中存储已经添加过的标签 此集合将约束被过滤的img列表 集合中的元素无法dup
    # print(" - 差分过滤开始处理")
    for i, cluster in enumerate(tqdm(lpips_clustering(images, thre), file=sys.stdout, desc=" - 差分过滤开始处理", ascii="░▒█")):  # 聚类方法 -1表示noise，与sklearn中的相同
        if cluster == -1:
            clustered_imgs.append(images[i])
        elif cluster not in added_clusters:
            clustered_imgs.append(images[i])
            added_clusters.add(cluster)
    gr.Info("差分过滤已结束")
    output_cache = clustered_imgs
    return get_output_status(output_cache)+"上次操作: 差分过滤"


def three_stage(dataset_name, rep_name=None):
    gr.Info("三阶分割开始处理")
    global output_cache
    if not dataset_name.endswith(' (kohya)'):
        if dataset_name.endswith("_processed"):
            process_dir = f"dataset/{dataset_name}"
        else:
            process_dir = f"dataset/{dataset_name}_processed"
        local_source = LocalSource(f"dataset/{dataset_name}")
        local_source.attach(
            ThreeStageSplitAction(),
        ).export(TextualInversionExporter(process_dir, True))
    else:
        if re.search(r'\d+_(.*)', rep_name) == 'processed':
            pass
        local_source = LocalSource(f'dataset/_kohya/{dataset_name.replace(" (kohya)", "")}/{rep_name}')
        repeat = int(re.search(r'(\d+)_.*', rep_name).group(1)//2)
        local_source.attach(
            ThreeStageSplitAction(),
        ).export(TextualInversionExporter(f'dataset/_kohya/{dataset_name.replace(" (kohya)", "")}/{str(repeat) if repeat != 0 else str(1)}_processed', True))

    gr.Info("三阶分割已结束")
    output_cache = []
    return "已保存至"+process_dir+"文件夹"

# def person_detect(dataset_name, level, version, max_infer_size, conf_threshold, iou_threshold):
#     global output_cache
#     images = dataset_getImg(dataset_name)[0]
#     detected = []
#     if level:
#         level = "m"
#     else:
#         level = "n"
#     print(" - 人物检测开始处理")
#     for img in tqdm(images):
#         detected.append(detect_person(img, level, version, max_infer_size, conf_threshold, iou_threshold))
#     output_cache = detected
#     return detected


def face_detect(dataset_name, level, version, max_infer_size, conf_threshold, iou_threshold, rep_name=None):
    global output_cache
    if has_image(output_cache):
        images = output_cache
        gr.Info("面部检测开始处理 <- 缓存")
    else:
        images = dataset_getImg(dataset_name, rep_name)[0]
        gr.Info("面部检测开始处理 <- 数据集")
    detected = []
    if level:
        level = "s"
    else:
        level = "n"
    # print(" - 面部检测开始处理")
    # print("   *将返回区域结果")
    for img in tqdm(images, file=sys.stdout, desc=" - 面部检测开始处理", ascii="░▒█"):
        detected.append(detect_faces(img, level, version, max_infer_size, conf_threshold, iou_threshold))
    gr.Info("面部检测已结束")
    output_cache = detected
    return get_output_status(output_cache)+"上次操作: 面部检测"


def head_detect(dataset_name, level, max_infer_size, conf_threshold, iou_threshold, rep_name=None):
    global output_cache
    if has_image(output_cache):
        images = output_cache
        gr.Info("头部检测开始处理 <- 缓存")
    else:
        images = dataset_getImg(dataset_name, rep_name)[0]
        gr.Info("头部检测开始处理 <- 数据集")
    detected = []
    if level:
        level = "s"
    else:
        level = "n"
    # print(" - 头部检测开始处理")
    # print("   *将返回区域结果")
    for img in tqdm(images, file=sys.stdout, ascii="░▒█", desc=" - 头部检测开始处理"):
        detected.append(detect_heads(img, level, max_infer_size, conf_threshold, iou_threshold))
    gr.Info("头部检测已结束")
    output_cache = detected
    return get_output_status(output_cache)+"上次操作: 头部检测"


def text_detect(dataset_name, rep_name=None):
    global output_cache
    if has_image(output_cache):
        images = output_cache
        gr.Info("文本检测开始处理 <- 缓存")
    else:
        images = dataset_getImg(dataset_name, rep_name)[0]
        gr.Info("文本检测开始处理 <- 数据集")
    detected = []
    for img in tqdm(images, file=sys.stdout, ascii="░▒█", desc=" - 文本检测开始处理"):
        detected.append(detect_text_with_ocr(img))
    gr.Info("文本检测已结束")
    output_cache = detected
    return get_output_status(output_cache)+"上次操作: 文本检测"


def area_fill(dataset_name, is_random, color, rep_name=None):
    global output_cache
    area = output_cache
    gr.Info("区域填充开始处理")
    images = dataset_getImg(dataset_name, rep_name)[0]
    fill = []
    xyxy = []
    # print(" - 区域填充开始处理")
    for img, xyxys in tzip(images, area, file=sys.stdout, ascii="░▒█", desc=" - 区域填充开始处理"):
        if xyxys:
            for exy in [xyxys][0]:
                xyxy.append(exy[0])
        if is_random:
            color = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
            color = random.choice(color)
            # color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        # print(img, eval(area), xyxys, xyxys[0][0])
        fill.append(censor_areas(img, 'color', xyxy, color=color))  # [xyxys[0][0]]
    # os.makedirs(f"processed/{dataset_name}", exist_ok=True)
    # for i, sv in enumerate(fill):
    #     sv.save(f"processed/{dataset_name}/{dataset_name}_AreaFill_{i+1}.png")
    gr.Info("区域填充已结束")
    output_cache = fill
    return get_output_status(output_cache)+"上次操作: 区域填充"


def area_blur(dataset_name, rad, rep_name=None):
    global output_cache
    area = output_cache
    gr.Info("区域模糊开始处理")
    images = dataset_getImg(dataset_name, rep_name)[0]
    blur = []
    xyxy = []
    for img, xyxys in tzip(images, area, file=sys.stdout, ascii="░▒█", desc=" - 区域模糊开始处理"):
        if xyxys:
            for exy in [xyxys][0]:
                xyxy.append(exy[0])
            blur.append(censor_areas(img, 'blur', xyxy, radius=rad))
        else:
            blur.append(img)
    output_cache = blur
    gr.Info("区域模糊已结束")
    return get_output_status(output_cache)+"上次操作: 区域模糊"


def crop_hw(dataset_name, rep_name=None):
    global output_cache
    mask_info = output_cache
    gr.Info("区域剪裁开始处理")
    images = dataset_getImg(dataset_name, rep_name)[0]
    result = []
    for img, infos in zip(images, mask_info):
        # infos = infos[0]
        # logger.debug(infos)
        for einfo in infos:
            (x0, y0, x1, y1) = einfo[0]
            detect_type = einfo[1]
            # score = infos[2]
            img_array = numpy.array(rgb_encode(img))
            h, w = img_array.shape[1:]
            mask = numpy.zeros((h, w))
            # print(mask, h, w)
            if detect_type == 'face' or detect_type == 'head' or detect_type == 'text':
                mask[y0:y1, x0:x1] = 1
                # print(mask)
            else:
                output_cache = []
                gr.Warning("区域剪裁: 当前运行结果不支持剪裁")
                return "此内容不支持剪裁"
            result.append(squeeze(img, mask))
    output_cache = result
    return get_output_status(output_cache)+"上次操作: 区域剪裁"


def crop_trans(dataset_name, threshold, filter_size, rep_name=None):
    global output_cache
    if has_image(output_cache):
        images = output_cache
        gr.Info("自适应剪裁开始处理 <- 缓存")
    else:
        images = dataset_getImg(dataset_name, rep_name)[0]
        gr.Info("自适应剪裁开始处理 <- 数据集")
    out = []
    for img in tqdm(images, file=sys.stdout, desc=" - 自适应剪裁开始处理", ascii="░▒█"):
        if img is not None:
            out.append(squeeze_with_transparency(img, threshold, filter_size))
    gr.Info("自适应剪裁已结束")
    output_cache = out
    return get_output_status(output_cache)+"上次操作: 自适应剪裁"


def img_segment(dataset_name, scale, rep_name=None):
    global output_cache
    if has_image(output_cache):
        images = output_cache
        gr.Info("人物分离开始处理 <- 缓存")
    else:
        images = dataset_getImg(dataset_name, rep_name)[0]
        gr.Info("人物分离开始处理 <- 数据集")
    out = []
    # print(" - 人物分离开始处理")
    for img in tqdm(images, file=sys.stdout, desc=" - 人物分离开始处理", ascii="░▒█"):
        out.append(segment_rgba_with_isnetis(img, scale)[1])  # mask信息被丢弃了
    gr.Info("人物分离已结束")
    output_cache = out
    return get_output_status(output_cache)+"上次操作: 人物分离"


def ref_datasets(need_list=False):
    # 遍历本地数据集
    list_datasets = []
    with os.scandir("dataset") as datasets:
        for each_dataset in datasets:
            if not each_dataset.name.startswith('.') and each_dataset.is_dir():
                if each_dataset.name == '_kohya':
                    with os.scandir(each_dataset) as kohya_datasets:
                        for kohya_dataset in kohya_datasets:
                            if not kohya_dataset.name.startswith('.') and kohya_dataset.is_dir():
                                list_datasets.append(kohya_dataset.name + ' (kohya)')
                else:
                    list_datasets.append(each_dataset.name)
    if need_list:
        return list_datasets
    else:
        gr.Info("数据集已更新")
        return gr.update(choices=list_datasets)


def ref_kohya_rep(kohya_dataset, need_list=False):
    list_reps = []
    if kohya_dataset.endswith(' (kohya)'):
        with os.scandir(f'dataset/_kohya/{kohya_dataset.replace(" (kohya)", "")}') as dataset:
            for each_rep in dataset:
                if re.match(r"\d+_", each_rep.name) and each_rep.is_dir():
                    list_reps.append(each_rep.name)
        if need_list:
            return list_reps
        else:
            return gr.update(choices=list_reps, visible=True, value=list_reps[0])
    else:
        if need_list:
            return []
        else:
            gr.Warning("数据集类型错误")
            return gr.update(choices=None)


def ref_customList(need_list=False):
    custom_blacklist = []
    with os.scandir("cfgs/blacklist") as blacklists:
        for each_black in blacklists:
            if not each_black.name.startswith('.') and each_black.is_file():
                if each_black.name.endswith('.txt') or each_black.name.endswith('.json'):
                    custom_blacklist.append(each_black.name)
    if need_list:
        return custom_blacklist
    else:
        gr.Info("标签黑名单已更新")
        return gr.update(choices=custom_blacklist)


def ref_runs(dataset_name, need_list=False):
    runs_list = []
    try:
        with os.scandir(f"runs/{dataset_name}/ckpts") as conv_list:
            for conv in conv_list:
                # print("遍历了一个conv")
                if conv.is_file():
                    if conv.name.endswith('.pt') and not dataset_name == '_kohya':
                        runs_list.append(conv.name.replace(dataset_name+"-", "").replace(".pt", ""))
    except FileNotFoundError:
        if need_list:
            return []
        else:
            return gr.update(choices=[])
    if runs_list is not None:
        runs_list = sorted(runs_list, key=int, reverse=True)
        if need_list:
            return runs_list
        else:
            gr.Info("训练结果已更新")
            # print("结果"+str(runs_list))
            return gr.update(choices=runs_list)


def run_train_lora(dataset_name, epoch, bs, toml_index, is_pipeline=False):
    logger.info("LoRA开始训练")
    gr.Info(f"[{dataset_name}] LoRA开始训练")
    if not is_pipeline:
        if not dataset_name.endswith(' (kohya)'):  # from dataset_dropdown
            raise DatasetTypeError(dataset_name, "正在尝试加载kohya数据集")
        else:
            r_dataset_name = dataset_name.replace(" (kohya)", "")
        kohya_train_lora(f"dataset/_kohya/{r_dataset_name}", r_dataset_name, f"runs/_kohya/{r_dataset_name}", epoch, bs, toml_index)
        for folder_name in os.listdir(f"dataset/_kohya/{r_dataset_name}"):
            if re.match(r"\d+_", folder_name):
                save_recommended_tags(f"dataset/_kohya/{r_dataset_name}/{folder_name}", r_dataset_name, f"runs/_kohya/{r_dataset_name}")
    else:
        r_dataset_name = dataset_name
        kohya_train_lora(f"pipeline/dataset/_kohya/{r_dataset_name}", r_dataset_name, f"pipeline/runs/_kohya/{r_dataset_name}", epoch, bs, toml_index)
        save_recommended_tags(f"pipeline/dataset/_kohya/{r_dataset_name}/1_{r_dataset_name}", r_dataset_name, f"pipeline/runs/_kohya/{r_dataset_name}")
    return "LoRA训练完成"


def convert_weights(dataset_name, step):
    global output_cache
    gr.Info("开始转换LoRA")
    # logging.try_init_root(logging.INFO)
    convert_to_webui_lora(f"runs/{dataset_name}/ckpts/unet-{step}.safetensors",
                          f"runs/{dataset_name}/ckpts/text_encoder-{step}.safetensors",
                          os.path.join(f"runs/{dataset_name}/ckpts", f"{dataset_name}-lora-{step}.safetensors")
                          )
    gr.Info("LoRA转换已结束")
    output_cache = []
    return "已执行转换"


def tagger_chooser_ctrl(evt: gr.SelectData):  # 此方法使用全局变量
    # print(evt.value+"正在选择")
    # 隐藏所有打标器设置
    updates = {}
    for tagger in taggers:
        # if tagger == "标签黑名单":
        #     tagger = "dropper"
        if tagger == "json解析":
            tagger = "anal"
        updates[globals()[f"tagger_{tagger}_settings"]] = gr.update(visible=False)
    # 显示打标器设置
    if evt.value in taggers:
        # if evt.value == "标签黑名单":
        #     evt.value = "dropper"
        if evt.value == "json解析":
            evt.value = "anal"
        updates[globals()[f"tagger_{evt.value}_settings"]] = gr.update(visible=True)
        # print(evt.value, tagger_dropper_settings.visible)
    # updates[globals()[f"wd14_use_blacklist"]] = gr.update(value=wd14_use_blacklist.value)
    # updates[globals()[f"ml_use_blacklist"]] = gr.update(value=ml_use_blacklist.value)
    # updates[globals()[f"tagger_dropper_settings"]] = gr.update(visible=tagger_dropper_settings.visible)
    return updates


def blacklist_settings_ctrl(evt: gr.SelectData):
    updates = {}
    if evt.selected:
        updates[tagger_dropper_settings] = gr.update(visible=True)
    else:
        updates[tagger_dropper_settings] = gr.update(visible=False)
    return updates


def kohya_rep_ctrl(evt: gr.SelectData):
    updates = {}
    if evt.value.endswith(' (kohya)'):
        updates[kohya_rep_dropdown] = ref_kohya_rep(evt.value)
        updates[ref_rep_button] = gr.update(visible=True)
    else:
        updates[kohya_rep_dropdown] = gr.update(visible=False)
        updates[ref_rep_button] = gr.update(visible=False)
    return updates


def custom_blacklist_ctrl(evt: gr.SelectData):
    if evt.selected:
        dropper_update = {globals()[f"drop_custom_setting"]: gr.update(visible=False)}
    else:
        dropper_update = {globals()[f"drop_custom_setting"]: gr.update(visible=True)}
    return dropper_update


def pixiv_setting_ctrl(evt: gr.SelectData):
    if evt.index == 1:
        update = {globals()[f"pixiv_settings"]: gr.update(visible=True)}
    else:
        update = {globals()[f"pixiv_settings"]: gr.update(visible=False)}
    return update


def color_picker_ctrl(evt: gr.SelectData):
    if evt.selected:
        update = {globals()[f"areaf_color"]: gr.update(visible=False)}
    else:
        update = {globals()[f"areaf_color"]: gr.update(visible=True)}
    return update


def save_output_ctrl():
    global output_cache
    if has_image(output_cache):
        update = {globals()[f"save_output"]: gr.update(interactive=True)}
    else:
        update = {globals()[f"save_output"]: gr.update(interactive=False)}
    return update


def saving_output(dataset_name):
    global output_cache
    count = 0
    if dataset_name.endswith("_processed"):
        process_dir = f"dataset/{dataset_name}"
    else:
        process_dir = f"dataset/{dataset_name}_processed"
    if has_image(output_cache):
        gr.Info("开始保存运行结果")
        os.makedirs(process_dir, exist_ok=True)
        anyfiles = os.listdir(process_dir)
        # print(" - 开始保存运行结果")
        for anyfile in anyfiles:
            os.remove(f"{process_dir}/{anyfile}")
        for i, sv in enumerate(tqdm(output_cache, file=sys.stdout, desc=" - 开始保存运行结果", ascii="░▒█")):
            sv.save(f"{process_dir}/{dataset_name}_{i+1}.png")
            count = count+1
        gr.Info("已保存"+str(count)+" 张图像至"+process_dir+"数据集")
        output_cache = []
        return "已保存 "+str(count)+" 张图像至"+process_dir+"数据集"
    else:
        gr.Warning("无法保存: 运行结果内没有图像")


def tagging_main(dataset_name, ttype, wd14_tagger, wd14_general_thre, wd14_character_thre, wd14_weight, wd14_overlap, ml_real_name, ml_thre, ml_scale, ml_weight, ml_ratio, ml_overlap, need_black, drop_presets, drop_custom, exists_txt, del_json, rep_name=None):
    global output_cache
    images = dataset_getImg(dataset_name, rep_name)[0]
    img_name = dataset_getImg(dataset_name, rep_name)[1]
    if ttype == taggers[0]:
        gr.Info("数据打标开始处理 打标器: wd14")
        for img, name in tzip(images, img_name, file=sys.stdout, ascii="░▒█", desc=" - 数据打标开始处理"):
            result = get_wd14_tags(img, wd14_tagger, wd14_general_thre, wd14_character_thre, wd14_overlap)
            if result[2]:
                result = tags_to_text(result[1], include_score=wd14_weight)+', '+tags_to_text(result[2], include_score=wd14_weight)  # features and chars
            else:
                result = tags_to_text(result[1], include_score=wd14_weight)
            if need_black:
                # print(result)
                result = str(str(drop_blacklisted_tags([result], drop_presets, drop_custom))[2:-2])
            if result:
                name = name.replace(".txt", "").replace(".jpg", "").replace(".png", "").replace(".jpeg", "")
                if os.path.isfile(f'dataset/{dataset_name}/{name}.txt'):
                    if exists_txt == "复制文件":
                        os.rename(f'dataset/{dataset_name}/{name}.txt', f'{dataset_name}/{name}_backup.txt')
                        with open(f'dataset/{dataset_name}/{name}.txt', 'w') as tag:
                            tag.write(result)
                    elif exists_txt == "忽略文件":
                        pass
                    elif exists_txt == "附加标签":
                        with open(f'dataset/{dataset_name}/{name}.txt', 'a+') as tag:
                            tag.write(result)
                    elif exists_txt == "覆盖文件":
                        with open(f'dataset/{dataset_name}/{name}.txt', 'w') as tag:
                            tag.write(result)
                else:
                    with open(f'dataset/{dataset_name}/{name}.txt', 'w') as tag:
                        tag.write(result)
        gr.Info("数据打标已结束")
    elif ttype == taggers[1]:
        gr.Info("数据打标开始处理 打标器: mldanbooru")
        # print(" - 数据打标开始处理")
        for img, name in tzip(images, img_name, file=sys.stdout, ascii="░▒█", desc=" - 数据打标开始处理"):
            result = get_mldanbooru_tags(img, ml_real_name, ml_thre, ml_scale, ml_ratio, ml_overlap)
            result = tags_to_text(result, include_score=ml_weight)
            if need_black:
                result = str(str(drop_blacklisted_tags([result], drop_presets, drop_custom))[2:-2])
            # print(result)
            if result:
                name = name.replace(".txt", "")
                if os.path.isfile(f'dataset/{dataset_name}/{name}.txt'):
                    if exists_txt == "复制文件":
                        os.rename(f'dataset/{dataset_name}/{name}.txt', f'{dataset_name}/{name}_backup.txt')
                        with open(f'dataset/{dataset_name}/{name}.txt', 'w') as tag:
                            tag.write(result)
                    elif exists_txt == "忽略文件":
                        pass
                    elif exists_txt == "附加标签":
                        with open(f'dataset/{dataset_name}/{name}.txt', 'a+') as tag:
                            tag.write(result)
                    elif exists_txt == "覆盖文件":
                        with open(f'dataset/{dataset_name}/{name}.txt', 'w') as tag:
                            tag.write(result)
                else:
                    with open(f'dataset/{dataset_name}/{name}.txt', 'w') as tag:
                        tag.write(result)
        gr.Info("数据打标已结束")
    elif ttype == taggers[2]:
        gr.Info("标签解析开始处理")
        json_files = glob.glob(f'dataset/{dataset_name}/.*.json')
        # print(" - 标签解析开始处理")
        for json_file in tqdm(json_files, file=sys.stdout, desc=" - 标签解析开始处理", ascii="░▒█"):
            with open(json_file, 'r') as f:
                jdata = json.load(f)
            danbooru_data = jdata.get('danbooru', {})
            tag_json_general = danbooru_data.get('tag_string_general', '')
            tag_json_character = danbooru_data.get('tag_string_character, ')
            if tag_json_general:
                tag_json_general = tag_json_general.replace(' ', ', ')
            if tag_json_character:
                tag_json_character = tag_json_character.replace(' ', ', ')
            if tag_json_general is None and tag_json_character is None:
                gr.Warning("标签解析: 数据集内无json标签")
                return get_output_status(output_cache)+"标签解析失败"
            elif tag_json_general is None:
                tag_json = tag_json_character
            elif tag_json_character is None:
                tag_json = tag_json_general
            else:
                tag_json = f"{tag_json_general}\n{tag_json_character}\n"
            if need_black:
                tag_json = str(str(drop_blacklisted_tags([tag_json], drop_presets, drop_custom))[2:-2])
            txtfile_name = json_file.replace('.', '', 1).replace('_meta.json', '.txt')
            if tag_json_general or tag_json_character:
                if os.path.isfile(f'{txtfile_name}'):
                    if exists_txt == "复制文件":
                        os.rename(f'{txtfile_name}', f'{txtfile_name}'.replace('.txt', '_backup.txt'))
                        with open(f'{txtfile_name}', 'w') as f:
                            f.write(tag_json)
                    elif exists_txt == "忽略文件":
                        pass
                    elif exists_txt == "附加标签":
                        with open(f'{txtfile_name}', 'a+') as f:
                            f.write(tag_json)
                    elif exists_txt == "覆盖文件":
                        with open(f'{txtfile_name}', 'w') as f:
                            f.write(tag_json)
                else:
                    with open(f'{txtfile_name}', 'w') as f:
                        f.write(tag_json)
                if del_json:
                    os.remove(json_file)
        gr.Info("标签解析已结束")


# @gr.StateHandler
def pre_rating_limit(rating):
    updates = {}
    # print(evt.index)
    # print(pre_rating)
    # print(rating)
    if not rating:
        updates[download_button] = gr.update(interactive=False)
    else:
        updates[download_button] = gr.update(interactive=True)
    return updates


# @gr.StateHandler
def illu_source_limit(i_source):
    updates = {}
    if not i_source:
        updates[illu_button] = gr.update(interactive=False)
    else:
        updates[illu_button] = gr.update(interactive=True)
    return updates


def save_settings(p_token, f_cookie, c_token, pro_ip, pro_port, pro_enabled, ver_enabled, thm_light, thm_style):
    global cfg
    cfg['pixiv_token'] = p_token
    cfg['fanbox_cookie'] = f_cookie
    cfg['civitai_token'] = c_token
    cfg['proxie_ip'] = pro_ip
    cfg['proxie_port'] = pro_port
    cfg['proxie_enabled'] = pro_enabled
    cfg['verify_enabled'] = ver_enabled
    cfg['theme_light'] = 'Light' if thm_light == '亮色' else 'Dark'
    cfg['theme_style'] = thm_style if not thm_style == '默认' else 'Default'
    with open('config.json', 'w') as f:
        json.dump(cfg, f, ensure_ascii=False, indent=4)
    gr.Info("设置已保存")
    load_settings()
    # 刷新设置页面
    return get_output_status(output_cache)+"设置已保存"


def load_settings():
    global cfg
    if os.path.getsize('config.json') > 0:  # 检查文件是否为空
        with open('config.json', 'r') as config:
            cfg = json.load(config)
    else:
        cfg = {}
    gr.Info("设置已读取")


def load_css():
    global cfg
    css_files = []
    merged_css = ""
    if cfg.get('theme_style', 'Default') == 'NovelAI':
        css_files.append('style/novel.css')
    css_files.append('style/apple.css')
    if css_files:
        for css_file in css_files:
            with open(css_file, "r") as f:
                css_content = f.read()
            merged_css += css_content
    logger.success("已加载css")
    return merged_css


def pixiv_login():
    global pyapi
    global cfg
    pyapi = AppPixivAPI()
    for _ in range(3):
        try:
            pyapi.auth(refresh_token=cfg.get('pixiv_token', ''))
            gr.Info("Pixiv已登录")
            logger.success("[信息] - Pixiv登录成功")
            break
        except PixivError:
            time.sleep(10)
        if not cfg.get('pixiv_token', ''):
            gr.Warning("Pixiv登录失败，因为没有设置访问令牌")
            logger.warning("[警告] - Pixiv登录失败，因为没有设置访问令牌")
            break
    else:
        gr.Warning("Pixiv登录失败")
        logger.warning("[警告] - Pixiv登录失败，已尝试三次，请前往设置检查刷新令牌，并尝试重新登录")


def pipeline_start_plora(ch_names):
    return pipeline_start(ch_names, 0)


def pipeline_start_lora(ch_names, toml_index):
    return pipeline_start(ch_names, 1, toml_index)


def pipeline_start(ch_names, train_type, toml_index=None):
    global output_cache
    global cfg
    bs = 4
    epoc = 10
    is_kohya = bool(train_type)
    riyu = kakasi()
    actions = [NoMonochromeAction(), CCIPAction(), PersonSplitAction(),  # ccip角色聚类
               HeadCountAction(1), TaggingAction(force=True),
               FilterSimilarAction('all'), ModeConvertAction('RGB'),  # FilterSimilar: lpips差分过滤
               FileExtAction(ext='.png'),  # png格式质量无损
               FirstNSelectAction(1000)]  # 700+
    ch_list = ch_names.split(',')
    for ch in ch_list:
        gr.Info(f"[{ch}]"+" 全自动训练开始")
        ch = ch.replace(' ', '_')
        ch_e = ''.join([r['hepburn']for r in riyu.convert(re.sub(r'[^\w\s()]', '', ''.join([word if not (u'\u4e00' <= word <= u'\u9fff') else lazy_pinyin(ch)[i] for i, word in enumerate(ch)])))]).replace(' ', '_')
        if not is_kohya:
            save_path = f"pipeline\\dataset\\{ch_e}"
        else:
            save_path = f"pipeline\\dataset\\_kohya\\{ch_e}\\1_{ch_e}"
###
        source_init = GcharAutoSource(ch, pixiv_refresh_token=cfg.get('pixiv_token', ''))
        source_init.attach(*actions).export(
            TextualInversionExporter(save_path)
        )
###
        if not is_kohya:
            run_train_plora(ch_e, bs=bs, epoc=epoc, min_step=2000, is_pipeline=True)  # bs, epoch 32 25
        else:
            run_train_lora(ch_e, bs=bs, epoch=epoc, toml_index=toml_index, is_pipeline=True)
###

        def huggingface(workdir: str, repository, revision, n_repeats, pretrained_model,
                        width, height, clip_skip, infer_steps):
            logging.try_init_root(logging.INFO)
            deploy_to_huggingface(
                workdir, repository, revision, n_repeats, pretrained_model,
                clip_skip, width, height, infer_steps, ds_dir=save_path, is_kohya=is_kohya
            )

        def rehf(repository, revision, n_repeats, pretrained_model,
                 width, height, clip_skip, infer_steps):
            from pathlib import Path
            logging.try_init_root(logging.INFO)
            with TemporaryDirectory() as workdir:
                logging.info(f'Downloading models for {workdir!r} ...')
                hf_fs = cyber_get_hf_fs()
                for f in tqdm(hf_fs.glob(f'{repository}/*/raw/*')):
                    rel_file = Path(os.path.relpath(f, repository)).as_posix()
                    local_file = os.path.join(workdir, 'ckpts', os.path.basename(rel_file))
                    if os.path.dirname(local_file):
                        os.makedirs(os.path.dirname(local_file), exist_ok=True)
                    cyber_download_file(
                        hf_hub_url(repository, filename=rel_file),
                        local_file
                    )

                logging.info(f'Regenerating tags for {workdir!r} ...')
                pt_name, _ = find_steps_in_workdir(workdir)
                game_name = pt_name.split('_')[-1]
                name = '_'.join(pt_name.split('_')[:-1])

                from gchar.games.dispatch.access import GAME_CHARS
                if game_name in GAME_CHARS:
                    ch_cls = GAME_CHARS[game_name]
                    ch = ch_cls.get(name)
                else:
                    ch = None

                if ch is None:
                    source = repository
                else:
                    source = ch

                logging.info(f'Regenerate tags for {source!r}, on {workdir!r}.')
                save_recommended_tags(source, name=pt_name, workdir=workdir)
                logging.info('Success!')

                deploy_to_huggingface(
                    workdir, repository, revision, n_repeats, pretrained_model,
                    clip_skip, width, height, infer_steps,
                )

        def civitai(repository, title, steps, epochs, draft, publish_time, allow_nsfw,
                    version_name, force_create, no_ccip_check, session, is_pipeline=False, is_kohya=False, verify=True):
            logging.try_init_root(logging.INFO)
            model_id = civitai_publish_from_hf(
                repository, title,
                step=steps, epoch=epochs, draft=draft,
                publish_at=publish_time, allow_nsfw_images=allow_nsfw,
                version_name=version_name, force_create_model=force_create,
                no_ccip_check=no_ccip_check, session=session, is_pipeline=is_pipeline,
                is_kohya=is_kohya, toml_index=toml_index, verify=verify,
            )
            url = f'https://civitai.com/models/{model_id}'
            if not draft:
                logging.info(f'Deploy success, model now can be seen at {url} .')
            else:
                logging.info(f'Draft created, it can be seed at {url} .')

        try:
            huggingface(workdir='pipeline/runs/' + ('_kohya/' if is_kohya else '') + ch_e, repository=None, n_repeats=3, pretrained_model=_DEFAULT_INFER_MODEL, width=512, height=768, clip_skip=2, infer_steps=30, revision='main')
        except Exception as e:
            logger.error(" - 错误:", e)
            raise e
        # if not is_kohya:
        #     try:
        #         rehf(repository=f'AppleHarem/{ch_e}', n_repeats=3, pretrained_model='_DEFAULT_INFER_MODEL', width=512, height=768, clip_skip=2, infer_steps=30, revision='main')
        #     except Exception as e:
        #         logger.error(" - 错误:", e)
        #         raise e
        try:
            civitai(repository=f'AppleHarem/{ch_e}', draft=False, allow_nsfw=True, force_create=False, no_ccip_check=False, session=None, epochs=epoc, publish_time=None, steps=None, title=f'{ch}/{ch_e}', version_name=None, is_pipeline=True, is_kohya=is_kohya, verify=cfg.get('verify_enabled', True))
        except Exception as e:
            logger.error(" - 错误:", e)
            raise e
        gr.Info(f"[{ch}]" + " 全自动训练完成")
        logger.success(" - 完成: 已完成"+ch+"角色上传")
    gr.Info("所有全自动训练任务完成")
    subprocess.call(["shutdown", "/s", "/t", "0"])  # TODO 自动关机功能
    return get_output_status(output_cache)+"所有任务完成"


def get_hf_token():
    return os.environ.get('HF_TOKEN')


def auto_crawler(chars_list, number):
    global crawler_clients
    crawler_clients["client_" + str(number)] = Client(f"AppleHarem/AppleBlock-{number}", hf_token=get_hf_token())
    logger.info(f"[信息] - 创建🐞{number}")
    crawler_clients["client_" + str(number)] = crawler_clients["client_" + str(number)].submit(get_hf_token(), chars_list, True, api_name="/crawlup")
    logger.info(f"[信息] - 提交🐞{number}训练集任务")
    gr.Info("[🐞"+str(number)+"] 全自动训练集任务已提交")


def auto_crawler_status(number):
    global crawler_clients
    if 'crawler_clients' in globals():
        client = crawler_clients.get("client_" + str(number))
        if client is not None:
            gr.Info(str(client.status()))
            logger.debug(str(client.status()))
            logger.debug(str(client.result()))
        else:
            gr.Warning("🐞"+str(number) + "尚未部署")
    else:
        gr.Warning("未部署任何🐞")


def auto_crawler_done(msg):
    logger.success(msg)
    gr.Info(msg)


def mirror_process():
    img_count = 0
    tag_count = 0
    gr.Info("选择包含图像的文件夹")
    root = tk.Tk()
    root.withdraw()
    pths = []
    while True:
        pth = filedialog.askdirectory()
        if pth:
            pths.append(os.path.abspath(pth))
        else:
            break
    gr.Info("快速镜像开始处理")
    for i_pth in pths:
        output_folder = i_pth + '_mirror'
        os.makedirs(output_folder, exist_ok=False)
        for filename in tqdm(os.listdir(i_pth), file=sys.stdout, desc=" - 快速镜像开始处理", ascii="░▒█"):
            if filename.endswith(".jpg") or filename.endswith(".png") or filename.endswith(".jpeg"):
                img_path = os.path.join(i_pth, filename)
                txt_file = os.path.splitext(img_path)[0] + ".txt"
                json_file = os.path.splitext(img_path)[0] + ".json"
                img = cv2.imread(img_path)
                img_mirror = cv2.flip(img, 1)
                cv2.imwrite(os.path.join(output_folder, filename), img_mirror)
                img_count = img_count + 1
                # tag
                if os.path.isfile(txt_file):
                    shutil.copy(txt_file, os.path.join(output_folder, os.path.basename(txt_file)))
                    tag_count = tag_count + 1
                if os.path.isfile(json_file):
                    shutil.copy(json_file, os.path.join(output_folder, os.path.basename(json_file)))
                    tag_count = tag_count + 1
    logger.success("快速镜像处理完成，输出位置与源文件夹位置相同")
    gr.Info("快速镜像处理完成")
    return get_output_status(output_cache)+"处理完毕, 共处理" + str(img_count) + "张图片, " + str(tag_count) + "个tag文件"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7862)
    parser.add_argument("--share", type=bool, default=False)
    args = parser.parse_args()

    matplotlib.use('Agg')
    # 读取配置文件
    global cfg
    load_settings()
    output_cache = []
    # cfg = {}
    # 登录pixiv
    global pyapi
    pixiv_login()
    crawler_clients = {}
    # 登录huggingface
    # hf_login(token=os.environ.get('HF_TOKEN'))
    # 主界面
    with gr.Blocks(css=load_css(), analytics_enabled=False) as iblock:
        quicksettings = gr.Row(elem_id="quicksettings")
        with quicksettings:
            dataset_dropdown = gr.Dropdown(ref_datasets(True), label="当前数据集", value=ref_datasets(True)[0], container=True, show_label=True, interactive=True, elem_id='dataset_dropbar')
            ref_datasets_button = gr.Button("🔄", elem_id='refresh_datasets')
            reps = ref_kohya_rep(dataset_dropdown.value, True)
            kohya_rep_dropdown = gr.Dropdown(reps, label="当前循环", value=reps[0] if reps else [], visible=False, elem_id='rep_dropbar', interactive=True, filterable=False)
            ref_rep_button = gr.Button("🔄", elem_id='refresh_reps', visible=False, interactive=True)
            dataset_dropdown.select(kohya_rep_ctrl, None, [kohya_rep_dropdown, ref_rep_button])
        with gr.Tab("数据获取"):
            with gr.Tab("图站"):
                source = gr.Radio(['Gelbooru', 'Pixiv', 'Zerochan', '自动'], label='选择图站', value='Gelbooru')
                char_name = gr.Textbox(label='角色名称', value='', placeholder='填入角色名')
                pre_min_size = gr.Textbox(label="最大尺寸", value="", placeholder="不填写将不生效", interactive=True)
                pre_background = gr.ColorPicker(label="背景色", value="#FFFFFF", interactive=True)
                pre_class = gr.CheckboxGroup(["素描", "3D"], label="风格过滤", value=None, type="index", interactive=True)
                pre_rating = gr.CheckboxGroup(["健全", "r15", "r18"], label="评级筛选", value=["健全"], type="index", interactive=True)
                pre_crop_person = gr.Checkbox(label="裁剪人物", value=False, interactive=True)
                pre_ccip_option = gr.Checkbox(label="特征匹配", value=False, interactive=True)
                pre_auto_tagging = gr.Checkbox(label="自动打标", value=False, interactive=True)
                with gr.Column(visible=False) as pixiv_settings:
                    pixiv_no_ai = gr.Checkbox(label="非AI生成", interactive=True, value=False)
                source.select(pixiv_setting_ctrl, None, [pixiv_settings])
                dl_count = gr.Textbox(label="下载数量", value='100', placeholder="无上限")
                # dl_count = gr.Slider(1, 1001, step=1, value=10, label="下载数量", elem_id='dl_count')
                # save_path = gr.Textbox(label='保存路径', value='dataset', placeholder='自动创建子文件夹')
                download_button = gr.Button("获取图片", variant="primary", interactive=True)
                with gr.Accordion("使用说明", open=False):
                    gr.Markdown("""对于单图站，填入要搜索的任何内容以获取对应标签图片\n
                                对于自动图站源，必须填入一个角色名\n
                                所有图站支持多内容顺序爬取，用半角逗号分隔，如\"铃兰,香风智乃\"\n
                                保存的图片会以搜索内容自动生成一个数据集，获取完成后刷新数据集即可查看\n
                                Pixiv源速度最慢、且质量最差""")
                pre_rating.change(pre_rating_limit, [pre_rating], [download_button])
            with gr.Tab("画师"):
                illu_name = gr.Textbox(label="画师名", placeholder="完整画师名")
                with gr.Row():
                    # illu_get_pixiv = gr.Checkbox(label="Pixiv", value=True, interactive=True)
                    # illu_get_fanbox = gr.Checkbox(label="Fanbox", value=False, interactive=True)
                    illu_get_source = gr.CheckboxGroup(["Pixiv", "Fanbox"], label="获取渠道", value=["Pixiv"], type="index", interactive=True)
                    illu_max_size = gr.Textbox(label="最大文件大小", info="MB", placeholder="不填写则无限制", value="16")
                illu_button = gr.Button("获取作品", variant="primary")
                with gr.Accordion("使用说明", open=False):
                    gr.Markdown("""仅支持pixiv fanbox 目前\n
                                关于完整画师名：要写画师在pixiv对应的名字，不可以写fanbox上的英文名""")
                illu_get_source.change(illu_source_limit, [illu_get_source], [illu_button])
                illu_getter_pic = gr.Image(type="filepath", label="到底是哪个画师?")
                # illu_getter_button = gr.Button("获取画师名", interactive=True)
                # illu_id_tmp = gr.Textbox(visible=False)
            # with gr.Tab("快速获取"):
            #     fast_tag = gr.Textbox(label="Tag", placeholder="aaa,bbb|ccc,ddd", value='')
            #     fast_button = gr.Button("开始获取", variant="primary", interactive=True)
        with gr.Tab("数据增强"):
            with gr.Tab("快速操作"):
                with gr.Accordion("三阶分割"):
                    stage_button = gr.Button("开始处理", variant="primary")
                with gr.Accordion("自适应剪裁"):
                    crop_trans_thre = gr.Slider(0.01, 1, label="容差阈值", value=0.7, step=0.01)
                    crop_trans_filter = gr.Slider(0, 10, label="羽化", value=5, step=1)
                    crop_trans_button = gr.Button("开始处理", variant="primary")
                    with gr.Accordion("使用说明", open=False):
                        gr.Markdown("""将数据集中的透明图片进行自适应剪裁。\n
                                    可对缓存或数据集进行操作。""")
                with gr.Accordion("差分过滤"):
                    cluster_threshold = gr.Slider(0, 1, label="阈值", step=0.1, value=0.45, interactive=True)
                    cluster_button = gr.Button("开始处理", variant="primary")
                    with gr.Accordion("使用说明", open=False):
                        gr.Markdown("""差分检测：LPIPS（感知图像补丁相似性） ，全称为Learned Perceptual Image Patch 
                                    Similarity，是一种用于评估图像相似性的度量方法。基于深度学习模型，通过比较图像之间的深度特征评估它们的相似性\n 
                                    LPIPS使用了预训练的分类网络（如AlexNet或VGG）来提取图像的特征。然后计算两个图像特征之间的余弦距离，
                                    并对所有层和空间维度的距离进行平均，可以得到一个值，用于表示两个图像之间的感知差异。\n
                                    *会暂存去除差分后的图片结果
                                    ![cluster](resource/lpips_full.plot.py.svg)""")
                with gr.Accordion("人物分离"):
                    seg_scale = gr.Slider(32, 2048, label="缩放大小", info="图像传递给模型时的缩放尺寸", step=32, value=1024, interactive=True)
                    with gr.Accordion("使用说明", open=False):
                        gr.Markdown("""人物分离\n
                                    *会暂存背景为透明的人物图片结果\n
                                    查阅skytnt的[复杂动漫抠像](https://github.com/SkyTNT/anime-segmentation/)""")
                    seg_button = gr.Button("开始处理", variant="primary")
                with gr.Accordion("快速操作说明", open=False):
                    gr.Markdown("""关于快速操作\n"
                                小苹果WebUI的设计理念是一个训练工具箱，用于执行轻量数据集的轻量操作，不支持过大数据集与部分极限任务\n
                                部分快速操作的结果将暂存到内存中，部分输入也会从内存结果中读取，而不是从源数据集中读取\n
                                这使你可以在UI中选择自己需要的工作流程""")
                    # TODO 未来将支持输入输出端点可视化
            with gr.Tab("区域检测"):
                # with gr.Accordion("人物检测"):
                #     ccip_level = gr.Checkbox(label="使用高精度", value=True, interactive=True)
                #     ccip_model = gr.Dropdown(["v0", "v1", "v1.1"], label="模型选择", value="v1.1", interactive=True)
                #     ccip_infer = gr.Slider(32, 2048, label="缩放大小", interactive=True, step=32, value=640, info="图像传递给模型时的缩放尺寸")
                #     ccip_conf = gr.Slider(0.01, 1, label="检测阈值", interactive=True, value=0.25, step=0.01, info="置信度高于此值的检测结果会被返回")
                #     ccip_iou = gr.Slider(0.01, 1, label="重叠阈值", interactive=True, value=0.7, step=0.01, info="重叠区域高于此阈值将会被丢弃")
                #     ccip_button = gr.Button("开始检测", variant="primary")
                #     with gr.Accordion("使用说明", open=False):
                #         gr.Markdown("角色检测：CCIP（对比角色图像预训练）从动漫角色图像中提取特征，计算两个角色之间的视觉差异，并确定两个图像是否"
                #                     "描绘相同的角色。![ccip](resource/ccip_full.plot.py.svg)"
                #                     "更多信息可查阅 [CCIP官方文档](https://deepghs.github.io/imgutils/main/api_doc/metrics/ccip.html).")
                with gr.Accordion("面部检测"):
                    faced_level = gr.Checkbox(value=True, label="使用高精度", interactive=True)
                    faced_model = gr.Dropdown(["v0", "v1", "v1.3", "v1.4"], label="模型选择", value="v1.4", interactive=True)
                    faced_infer = gr.Slider(32, 2048, label="缩放大小", interactive=True, step=32, value=640, info="图像传递给模型时的缩放尺寸")
                    faced_conf = gr.Slider(0.01, 1, label="检测阈值", interactive=True, value=0.25, step=0.01, info="置信度高于此值的检测结果会被返回")
                    faced_iou = gr.Slider(0.01, 1, label="重叠阈值", interactive=True, value=0.7, step=0.01, info="重叠区域高于此阈值将会被丢弃")
                    with gr.Accordion("使用说明", open=False):
                        gr.Markdown("""##面部检测
                                    来自imgutils检测模块
                                    ###此功能会返回一个区域结果，而不是图片结果""")
                    faced_button = gr.Button("开始检测", variant="primary")
                with gr.Accordion("头部检测"):
                    headd_level = gr.Checkbox(value=True, label="使用高精度", interactive=True)
                    headd_infer = gr.Slider(32, 2048, label="缩放大小", interactive=True, step=32, value=640, info="图像传递给模型时的缩放尺寸")
                    headd_conf = gr.Slider(0.01, 1, label="检测阈值", interactive=True, value=0.25, step=0.01, info="置信度高于此值的检测结果会被返回")
                    headd_iou = gr.Slider(0.01, 1, label="重叠阈值", interactive=True, value=0.7, step=0.01, info="重叠区域高于此阈值将会被丢弃")
                    with gr.Accordion("使用说明", open=False):
                        gr.Markdown("""##头部检测
                                    来自imgutils检测模块
                                    ###此功能会返回一个区域结果，而不是图片结果)""")
                    headd_button = gr.Button("开始检测", variant="primary")
                with gr.Accordion("文本检测"):
                    with gr.Accordion("使用说明", open=False):
                        gr.Markdown("""文本检测\n
                                    用ocr的方式检测文本的模块\n
                                    此功能会返回一个区域结果，而不是图片结果\n
                                    此功能结果质量差，不建议使用""")
                    textd_button = gr.Button("开始检测", variant="primary")
                with gr.Accordion("区域检测说明", open=False):
                    gr.Markdown("""此选项卡中的操作是检测操作\n
                                可接受结果中的图像，将暂存区域信息\n""")
            with gr.Tab("区域处理"):
                with gr.Accordion("区域填充"):
                    areaf_isRandom = gr.Checkbox(label="随机颜色", value=True, interactive=True)
                    areaf_color = gr.ColorPicker(label="自定义颜色", value="#00FF00", visible=not areaf_isRandom.value)
                    areaf_button = gr.Button("开始处理", variant="primary")
                    with gr.Accordion("使用说明", open=False):
                        gr.Markdown("""接收输出后的结果进行打码。\n
                                    运行结果内有区域信息，才可以填充...""")
                    areaf_isRandom.select(color_picker_ctrl, None, [areaf_color])
                with gr.Accordion("区域模糊"):
                    areab_radius = gr.Slider(1, 20, label="模糊强度", value=4, interactive=True, step=1)
                    areab_button = gr.Button("开始处理", variant="primary")
                    with gr.Accordion("使用说明", open=False):
                        gr.Markdown("""接收输出后的结果进行打码。\n
                                    运行结果内有区域信息，才可以模糊...""")
                with gr.Accordion("区域剪裁"):
                    crop_hw_button = gr.Button("开始处理", variant="primary")
                    with gr.Accordion("使用说明", open=False):
                        gr.Markdown("""将运行结果中的区域进行剪裁。\n
                                    运行结果内有区域信息，才可以剪裁...""")
            with gr.Tab("快捷工具"):
                with gr.Accordion("快速镜像"):
                    mirror_pickup = gr.Button("选择文件夹", variant="primary")
                    with gr.Accordion("使用说明", open=False):
                        gr.Markdown("""可选择多个文件夹，直到手动取消\n"
                                    "程序将自动帮你处理所有图像的镜像操作以及标签文件\n""")
                    with gr.Accordion("快捷工具说明", open=False):
                        gr.Markdown("""此类工具部分是为kohya设计的\n
                                    由于kohya数据集结构特殊，我们无法直接读取和处理kohya数据集的内容\n
                                    此类工具大部分使用了os库，因此你可以用它们处理计算机上任何位置的内容""")
        with gr.Tab("打标器"):
            taggers = ["wd14", "mldanbooru", "json解析"]
            tagger_type = gr.Dropdown(taggers, value=taggers[0], label="打标器", allow_custom_value=False, interactive=True, filterable=False)
            with gr.Column(visible=tagger_type.value == taggers[0]) as tagger_wd14_settings:
                wd14_tagger_model = gr.Dropdown(["SwinV2", "ConvNext", "ConvNextV2", "ViT", "MOAT"], value="ConvNextV2", label="打标模型", interactive=True, filterable=False)
                wd14_general_threshold = gr.Slider(0.01, 1, value=0.35, label="普通标签阈值", step=0.01, interactive=True)
                wd14_character_threshold = gr.Slider(0.01, 1, value=0.85, label="角色标签阈值", step=0.01, interactive=True)
                wd14_format_weight = gr.Checkbox(label="写入权重", value=False, interactive=True)
                wd14_drop_overlap = gr.Checkbox(value=True, label="精确打标", interactive=True)
                # wd14_use_blacklist = gr.Checkbox(label="使用黑名单", value=True, interactive=True)
            with gr.Column(visible=tagger_type.value == taggers[1]) as tagger_mldanbooru_settings:
                ml_use_real_name = gr.Checkbox(value=False, label="标签重定向", info="由于在Deepdanbooru训练后，Danbooru网站上的许多标签需要重命名和重定向，因此在某些应用场景中可能有必要使用最新的标签名称。")
                ml_threshold = gr.Slider(0.01, 1, value=0.7, label="标签阈值", step=0.01, interactive=True)
                ml_size = gr.Slider(32, 1024, value=448, step=32, label="缩放大小", interactive=True, info="将缩放后的图像传递给模型时的大小")
                ml_keep_ratio = gr.Checkbox(value=False, label="保持比例", info="保持训练集图像的原始比例", interactive=True)
                ml_format_weight = gr.Checkbox(label="写入权重", value=False, interactive=True)
                ml_drop_overlap = gr.Checkbox(value=True, label="精确打标", interactive=True)
                # ml_use_blacklist = gr.Checkbox(label="使用黑名单", value=True, interactive=True)
            with gr.Column(visible=tagger_type.value == taggers[2]) as tagger_anal_settings:
                with gr.Accordion("使用说明", open=False):
                    gr.Markdown("""用此脚本获取的图片附有json文件\n
                                "使用此打标器以从中提取tag\n"
                                "此功能不会检查图片，而是从所有可能的json文件中提取tag""")
                anal_del_json = gr.Checkbox(value=False, label="删除json", interactive=True)
            use_blacklist = gr.Checkbox(label="使用黑名单", value=True, interactive=True)
            with gr.Column(visible=use_blacklist.value) as tagger_dropper_settings:
                drop_use_presets = gr.Checkbox(value=True, label="使用在线黑名单", info="获取在线黑名单，来自alea31435", interactive=True)
                with gr.Column(visible=not drop_use_presets.value, elem_id="drop_custom_setting") as drop_custom_setting:
                    drop_custom_list = gr.Dropdown(ref_customList(True), value=ref_customList(True)[0], label="自定义黑名单", elem_id="custom_list", interactive=True, info="黑名单路径cfgs/blacklist/")
                    drop_ref_button = gr.Button("🔄", elem_id='refresh_custom_list')
            op_exists_txt = gr.Dropdown(["复制文件", "忽略文件", "覆盖文件", "附加标签"], value="附加标签", info="对于已存在标签，打标器的行为", show_label=False, interactive=True, filterable=False)
            tagger_button = gr.Button("打标", variant="primary")
            # tagger_type.select(tagger_chooser_ctrl, None, [globals()[f'tagger_{("dropper" if tagger == "标签黑名单" else tagger)}_settings'] for tagger in taggers])
            tagger_type.select(tagger_chooser_ctrl, None, [globals()[f'tagger_{("anal" if tagger == "json解析" else tagger)}_settings'] for tagger in taggers])
            # wd14_use_blacklist.select(blacklist_settings_ctrl, None, [tagger_dropper_settings])
            # ml_use_blacklist.select(blacklist_settings_ctrl, None, [tagger_dropper_settings])
            use_blacklist.select(blacklist_settings_ctrl, None, [tagger_dropper_settings])
            drop_use_presets.select(custom_blacklist_ctrl, None, [drop_custom_setting])
        with gr.Tab("PLoRA训练"):
            with gr.Tab("快速训练"):
                plora_min_step = gr.Textbox(label="最小步数", value='', placeholder='不填写将自动计算')
                plora_epoch = gr.Slider(1, 100, label="Epoch", value=10)
                plora_batch_size = gr.Slider(1, 64, label="Batch Size", value=4, step=1)
                plora_train_button = gr.Button("开始训练", variant="primary")
                with gr.Accordion("使用说明", open=False):
                    gr.Markdown("""训练详细说明..什么的""")
            with gr.Tab("高级训练"):
                with gr.Accordion("使用说明", open=False):
                    gr.Markdown("soon...")
            with gr.Tab("全自动训练") as tab_pipeline_plora:
                pipeline_text_plora = gr.Textbox(label="角色名称", placeholder="《输入角色名然后你的模型就出现在c站了》", info="要求角色名 用,分隔")
                pipeline_button_plora = gr.Button("开始全自动训练", variant="primary")
                with gr.Accordion("使用说明", open=False):
                    gr.Markdown("""《输入角色名然后你的模型就出现在c站了》\n
                                需要在设置中设置c站token\n
                                需要在计算机中添加环境变量: 键名 HF_TOKEN 值: 从登录的HuggingFace网站获取 在账号设置中创建访问令牌""")
        toml_presets = ['默认', '一杯哈萨姆', '琥珀青葉']
        with gr.Tab("LoRA训练"):
            with gr.Tab("快速训练"):
                lora_epoch = gr.Slider(1, 100, label="Epoch", value=10)
                lora_batch_size = gr.Slider(1, 64, label="Batch Size", value=1, step=1)
                lora_toml_presets = gr.Radio(toml_presets, label="参数", info="通用化的参数预设", type="index", value="默认", interactive=True)
                lora_train_button = gr.Button("开始训练", variant="primary")
            with gr.Tab("高级训练"):
                with gr.Accordion("使用说明", open=False):
                    gr.Markdown("soon...")
            with gr.Tab("全自动训练") as tab_pipeline_lora:
                pipeline_text_lora = gr.Textbox(label="角色名称", placeholder="《输入角色名然后你的模型就出现在c站了》", info="要求角色名 用,分隔")
                pipeline_toml_presets = gr.Radio(toml_presets, label="参数", info="通用化的参数预设", type="index", value="默认")
                pipeline_button_lora = gr.Button("开始全自动训练", variant="primary")
                with gr.Accordion("使用说明", open=False):
                    gr.Markdown("""《输入角色名然后你的模型就出现在c站了》\n
                                需要在设置中设置c站token\n
                                需要在计算机中添加环境变量: 键名 HF_TOKEN 值: 从登录的HuggingFace网站获取 在账号设置中创建访问令牌""")
        with gr.Tab("质量评估"):
            with gr.Accordion("使用说明", open=False):
                gr.Markdown("soon...")
        with gr.Tab("上传权重"):
            with gr.Accordion("LoRA合并", open=True):
                with gr.Column(elem_id="convert_lora_steps") as convert_lora_steps:
                    convert_step = gr.Dropdown(ref_runs(dataset_dropdown.value, True), value=ref_runs(dataset_dropdown.value, True)[0] if ref_runs(dataset_dropdown.value, True) else [], label="步数",
                                               info="HCP可用,合并对应步数的权重文件", elem_id="convert_list", multiselect=False, interactive=True, filterable=False)
                    convert_ref_button = gr.Button("🔄", elem_id='convert_ref_button')
                convert_weights_button = gr.Button("开始合并", variant="primary")
            with gr.Accordion("权重拆解", open=True):
                # 大模型权重拆解 sd 2 diffusers 格式
                pass
            with gr.Accordion("使用说明", open=False):
                gr.Markdown("上传权重到抱脸和C站 soon..")
        with gr.Tab("全自动数据集"):
            with gr.Tab("1机"):
                auto_crawl_1_chars = gr.Textbox(label="角色名称", placeholder="《输入角色名然后你的数据集就出现在抱脸了》", info="要求角色名 用,分隔")
                auto_crawl_1_button = gr.Button("开始全自动数据集", variant="primary")
                auto_crawl_1_status = gr.Button("查询状态")
                auto_crawl_1_number = gr.Textbox(value="1", visible=False)
            with gr.Tab("2机"):
                auto_crawl_2_chars = gr.Textbox(label="角色名称", placeholder="《输入角色名然后你的数据集就出现在抱脸了》", info="要求角色名 用,分隔")
                auto_crawl_2_button = gr.Button("开始全自动数据集", variant="primary")
                auto_crawl_2_status = gr.Button("查询状态")
                auto_crawl_2_number = gr.Textbox(value="2", visible=False)
            with gr.Tab("3机"):
                auto_crawl_3_chars = gr.Textbox(label="角色名称", placeholder="《输入角色名然后你的数据集就出现在抱脸了》", info="要求角色名 用,分隔")
                auto_crawl_3_button = gr.Button("开始全自动数据集", variant="primary")
                auto_crawl_3_status = gr.Button("查询状态")
                auto_crawl_3_number = gr.Textbox(value="3", visible=False)
            with gr.Accordion("使用说明", open=False):
                gr.Markdown("""《输入角色名然后你的数据集就出现在抱脸了》\n
                            需要设置抱脸token\n
                            你必须拥有组织的读写权限""")
        with gr.Tab("设置"):
            with gr.Tab("Pixiv"):
                pixiv_token = gr.Textbox(label="刷新令牌", placeholder="不填写将无法访问Pixiv", interactive=True, value=cfg.get('pixiv_token', ''))
                pixiv_get_token = gr.Button("前往查询", interactive=True)
                with gr.Accordion("令牌说明", open=False):
                    gr.Markdown("""获取Pixiv图片需要刷新令牌\n
                                用法：点击`前往获取`，将打开Pixiv网页，按F12启用开发者控制台，选择`网络/Network`，点击左侧第三个按钮`筛选器`，
                                筛选`callback?`点击继续使用此账号登录，此时页面会跳转，开发者控制台会出现一条请求，点击它，进入`标头`
                                复制`code=`后的内容，填入后台（黑窗口）按回车，后台将返回你的refresh token\n
                                打开webui时会尝试自动登录，如果失败请尝试下方登录按钮，需要先填写刷新令牌并保存\n
                                控制台中可以看到登录信息\n
                                取消查询请在后台按ctrl+c""")
                # settings_list = [pixiv_token]
                pixiv_manual_login = gr.Button("尝试登录", interactive=True)
            with gr.Tab("Fanbox"):
                fanbox_cookie = gr.Textbox(label="Cookie", lines=13, placeholder="不填写将无法获取Fanbox内容", interactive=True, value=cfg.get('fanbox_cookie', ''))
                fanbox_get_cookie = gr.Button("前往查询", interactive=True)
                with gr.Accordion("Cookie说明", open=False):
                    gr.Markdown("""获取Fanbox图片需要Kemono网站Cookie\n
                                Cookie格式：[{xxx},{x..}]，名为session的cookie\n
                                具体操作：使用EditThisCookie浏览器扩展\n
                                进入Kemono网站，导出cookie，将cookie粘贴到设置中，删除第一项和第三项，\n
                                无需[]大括号，只保留名为session的cookie{xxx}即可""")
            with gr.Tab("Civitai"):
                civitai_token = gr.Textbox(label="Cookie", lines=13, placeholder="不填写无法自动上传c站", interactive=True, value=cfg.get('civitai_token', ''))
            with gr.Tab("Huggingface"):
                hf_token_show = gr.Textbox(label="Token", value=get_hf_token(), info="Huggingface的token需要在环境变量中设置", interactive=False)
                hf_token_ref = gr.Button("刷新token")
            with gr.Tab("网络设置"):
                proxie_ip = gr.Textbox(label="代理IP地址", placeholder="代理软件的IP地址", value=cfg.get('proxie_ip', ''))
                proxie_port = gr.Textbox(label="代理端口", placeholder="代理软件中的端口", value=cfg.get('proxie_port', ''))
                proxie_enabled = gr.Checkbox(label="启用代理", interactive=True, value=cfg.get('proxie_enabled', False))
                verify_enabled = gr.Checkbox(label="启用验证", info="SSL/TLS 证书验证", value=cfg.get('verify_enabled', True))
            with gr.Tab("界面设置"):
                theme_light = gr.Radio(['亮色', '暗色'], label="颜色切换", interactive=True, info="需要重启", value='亮色' if cfg.get('theme_light', 'Light') == 'Light' else '暗色')
                theme_style = gr.Dropdown(['默认', 'NovelAI', 'Soft'], label="界面主题", interactive=True, info="需要重启", value='默认' if cfg.get('theme_style', 'Default') == 'Default' else cfg.get('theme_style', 'Default'), filterable=False)
            setting_save_button = gr.Button("保存", interactive=True, variant="primary")
            with gr.Accordion("使用说明", open=False):
                gr.Markdown("""###我只是个打酱油的...""")
        with gr.Column(elem_id="output"):
            message_output = gr.Textbox(label='运行结果', elem_id="message_output")
            save_output = gr.Button("💾", elem_id="save_output", interactive=False)
            message_output.change(save_output_ctrl, [], save_output)
        # dl_count.change(None, )
        mirror_pickup.click(mirror_process, [], [message_output])
        auto_crawl_1_button.click(auto_crawler, [auto_crawl_1_chars, auto_crawl_1_number], [])
        auto_crawl_2_button.click(auto_crawler, [auto_crawl_2_chars, auto_crawl_2_number], [])
        auto_crawl_3_button.click(auto_crawler, [auto_crawl_3_chars, auto_crawl_3_number], [])
        auto_crawl_1_status.click(auto_crawler_status, [auto_crawl_1_number], [])
        auto_crawl_2_status.click(auto_crawler_status, [auto_crawl_2_number], [])
        auto_crawl_3_status.click(auto_crawler_status, [auto_crawl_3_number], [])
        pipeline_button_plora.click(pipeline_start_plora, [pipeline_text_plora], [message_output])
        pipeline_button_lora.click(pipeline_start_lora, [pipeline_text_lora, pipeline_toml_presets], [message_output])
        setting_save_button.click(save_settings, [pixiv_token, fanbox_cookie, civitai_token, proxie_ip, proxie_port, proxie_enabled, verify_enabled, theme_light, theme_style], [message_output])
        pixiv_manual_login.click(pixiv_login, [], [])
        pixiv_get_token.click(get_ref_token, [], [])
        fanbox_get_cookie.click(get_fanbox_cookie, [], [])
        # fast_button.click(get_danbooru_fast, [fast_tag], [])
        # illu_getter_button.click(illu_getter, [illu_getter_pic], [message_output, illu_name])
        illu_getter_pic.upload(illu_getter, [illu_getter_pic], [message_output, illu_name])
        download_button.click(download_images, [source, char_name, pre_min_size, pre_background, pre_class, pre_rating, pre_crop_person, pre_ccip_option, pre_auto_tagging, dl_count, pixiv_no_ai],
                              [message_output], scroll_to_output=True)
        ref_datasets_button.click(ref_datasets, [], [dataset_dropdown])
        ref_rep_button.click(ref_kohya_rep, [dataset_dropdown], [kohya_rep_dropdown])
        stage_button.click(three_stage, [dataset_dropdown, kohya_rep_dropdown], [message_output])
        drop_ref_button.click(ref_customList, [], [drop_custom_list])
        convert_ref_button.click(ref_runs, [dataset_dropdown], [convert_step])
        convert_weights_button.click(convert_weights, [dataset_dropdown, convert_step], [message_output])
        cluster_button.click(clustering, [dataset_dropdown, cluster_threshold, kohya_rep_dropdown], [message_output], scroll_to_output=True)
        seg_button.click(img_segment, [dataset_dropdown, seg_scale, kohya_rep_dropdown], [message_output], scroll_to_output=True)
        # ccip_button.click(person_detect, [dataset_dropdown, ccip_level, ccip_model, ccip_infer, ccip_conf, ccip_iou], [message_output])
        faced_button.click(face_detect, [dataset_dropdown, faced_level, faced_model, faced_infer, faced_conf, faced_iou, kohya_rep_dropdown], [message_output], scroll_to_output=True)
        headd_button.click(head_detect, [dataset_dropdown, headd_level, headd_infer, headd_conf, headd_iou, kohya_rep_dropdown], [message_output], scroll_to_output=True)
        textd_button.click(text_detect, [dataset_dropdown, kohya_rep_dropdown], [message_output], scroll_to_output=True)
        plora_train_button.click(run_train_plora, [dataset_dropdown, plora_min_step, plora_batch_size, plora_epoch], [message_output], scroll_to_output=True)
        lora_train_button.click(run_train_lora, [dataset_dropdown, lora_epoch, lora_batch_size, lora_toml_presets], [message_output], scroll_to_output=True)
        areaf_button.click(area_fill, [dataset_dropdown, areaf_isRandom, areaf_color, kohya_rep_dropdown], [message_output], scroll_to_output=True)
        areab_button.click(area_blur, [dataset_dropdown, areab_radius, kohya_rep_dropdown], [message_output], scroll_to_output=True)
        crop_hw_button.click(crop_hw, [dataset_dropdown, kohya_rep_dropdown], [message_output], scroll_to_output=True)
        crop_trans_button.click(crop_trans, [dataset_dropdown, crop_trans_thre, crop_trans_filter, kohya_rep_dropdown], [message_output], scroll_to_output=True)
        tagger_button.click(tagging_main,
                            [dataset_dropdown, tagger_type, wd14_tagger_model, wd14_general_threshold, wd14_character_threshold, wd14_format_weight, wd14_drop_overlap, ml_use_real_name, ml_threshold,
                             ml_size, ml_format_weight, ml_keep_ratio, ml_drop_overlap, use_blacklist, drop_use_presets, drop_custom_list, op_exists_txt, anal_del_json, kohya_rep_dropdown], [message_output],
                            scroll_to_output=True)
        illu_button.click(download_illust, [illu_name, illu_get_source, illu_max_size], [message_output], scroll_to_output=True)
        save_output.click(saving_output, [dataset_dropdown], [message_output])
        iblock.title = "小苹果webui"

    # log.info(f"Server started at http://{args.host}:{args.port}")
    if sys.platform == "win32":
        webbrowser.open(f"http://{args.host}:{args.port}" + ("?__theme=dark" if cfg.get('theme_light', 'Light') == 'Dark' else ""))
    iblock.queue()
    iblock.launch(server_port=args.port, server_name=args.host, share=args.share)
