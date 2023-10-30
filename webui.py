import io
import logging
import sys
import matplotlib
import os
import gradio as gr
import random
import json
import glob
import numpy
from get_pixiv_ref_token import get_ref_token
from typing import Literal, cast
from tqdm import tqdm
from tqdm.contrib import tzip
from waifuc.action import HeadCountAction, AlignMinSizeAction, CCIPAction, ThreeStageSplitAction, ModeConvertAction, ClassFilterAction, PersonSplitAction, TaggingAction, RatingFilterAction, NoMonochromeAction
from waifuc.export import SaveExporter
from waifuc.source import DanbooruSource, PixivSearchSource, ZerochanSource, LocalSource
from PIL import Image
from train import run_train_plora
from imgutils.data import load_image, load_images, rgb_encode, rgb_decode
from imgutils.tagging import get_wd14_tags, get_mldanbooru_tags, drop_blacklisted_tags, drop_overlap_tags, tags_to_text
from imgutils.metrics import ccip_difference, ccip_clustering, lpips_clustering
from imgutils.operate import censor_areas, squeeze, squeeze_with_transparency
from imgutils.detect import detect_faces, detect_heads, detection_visualize, detect_person
from imgutils.segment import segment_rgba_with_isnetis
from cyberharem.publish.convert import convert_to_webui_lora

matplotlib.use('Agg')


def download_images(source_type, character_name, p_min_size, p_background, p_class, p_rating, p_crop_person, p_auto_tagging, num_images, ptoken, p_ai):
    global output_cache
    character_name = character_name.replace(' ', '_')  # 将空格替换为下划线
    save_path = 'dataset/' + character_name
    print("\n - 开始获取数据集")
    if source_type == 'Danbooru':
        source_init = DanbooruSource([character_name, 'solo'])
    elif source_type == 'Pixiv':
        source_init = PixivSearchSource(
            character_name,
            no_ai=p_ai,
            refresh_token=ptoken
        )
        source_init.attach(CCIPAction(source_init))  # 用于过滤pixiv传回的不相关角色图像
    elif source_type == 'Zerochan':
        source_init = ZerochanSource([character_name, 'solo'])
    else:
        output_cache = []
        return "图站错误"
    rating_map = {0: 'safe', 1: 'r15', 2: 'r18'}
    class_map = {1: 'illustration', 2: 'bangumi'}
    ratings_to_filter = set(rating_map.values()) - set([rating_map[i] for i in p_rating if i in rating_map])

    if p_class:
        if 0 in p_class:
            source_init.attach(NoMonochromeAction())
        class_to_filter = set(class_map.values()) - set([class_map[i] for i in p_class if i in class_map])
        source_init.attach(ClassFilterAction(cast(list[Literal['illustration', 2: 'bangumi']], list(ratings_to_filter))), RatingFilterAction(cast(list[Literal['safe', 'r15', 'r18']], list(class_to_filter))))  # 过滤具有评级的图片
    if p_crop_person:
        source_init.attach(PersonSplitAction())
    if p_auto_tagging:
        source_init.attach(TaggingAction(force=True))
    source_init.attach(
        AlignMinSizeAction(int(p_min_size)),
        ModeConvertAction('RGB', p_background),

        HeadCountAction(1),  # 只保留一头的图片
    )[:num_images].export(  # 只下载前num_images张图片
        SaveExporter(save_path)  # 将图片保存到指定路径
    )
    print(ratings_to_filter)
    output_cache = []
    return "已将获取图片保存至" + save_path


def dataset_getImg(dataset_name):  # 请确保每个方法中只调用一次 由于tqdm
    global output_cache
    # print(" - 开始加载图像")
    dataset_path = "dataset/" + dataset_name
    images = []
    img_name = []
    for filename in tqdm(os.listdir(dataset_path), file=sys.stdout, desc=" - 开始加载图像"):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            img = Image.open(os.path.join(dataset_path, filename))
            if img is not None:
                images.append(img)
                img_name.append(filename)
    # output_cache = images
    images = load_images(images)
    return images, img_name


def has_image(got_list):
    if any(isinstance(item, Image.Image) for item in got_list):
        return True
    else:
        return False


def clustering(dataset_name):
    global output_cache
    images = dataset_getImg(dataset_name)[0]
    # print(clusters)
    clustered_imgs = []
    added_clusters = set()  # 创建一个集合 其中存储已经添加过的标签 此集合将约束被过滤的img列表 集合中的元素无法dup
    # print(" - 差分过滤开始处理")
    for i, cluster in enumerate(tqdm(lpips_clustering(images), file=sys.stdout, desc=" - 差分过滤开始处理", ascii="░▒█")):  # 聚类方法 -1表示noise，与sklearn中的相同
        if cluster == -1:
            clustered_imgs.append(images[i])
        elif cluster not in added_clusters:
            clustered_imgs.append(images[i])
            added_clusters.add(cluster)
    output_cache = clustered_imgs
    return clustered_imgs


def three_stage(dataset_name):
    global output_cache
    if dataset_name.endswith("_processed"):
        process_dir = f"dataset/{dataset_name}"
    else:
        process_dir = f"dataset/{dataset_name}_processed"
    local_source = LocalSource(f"dataset/{dataset_name}")
    local_source.attach(
        ThreeStageSplitAction(),
    ).export(SaveExporter(process_dir, True))
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


def face_detect(dataset_name, level, version, max_infer_size, conf_threshold, iou_threshold):
    global output_cache
    images = dataset_getImg(dataset_name)[0]
    detected = []
    if level:
        level = "s"
    else:
        level = "n"
    # print(" - 面部检测开始处理")
    # print("   *将返回区域结果")
    for img in tqdm(images, file=sys.stdout, desc=" - 面部检测开始处理", ascii="░▒█"):
        detected.append(detect_faces(img, level, version, max_infer_size, conf_threshold, iou_threshold))
    output_cache = detected
    return detected


def head_detect(dataset_name, level, max_infer_size, conf_threshold, iou_threshold):
    global output_cache
    images = dataset_getImg(dataset_name)[0]
    detected = []
    if level:
        level = "s"
    else:
        level = "n"
    # print(" - 头部检测开始处理")
    # print("   *将返回区域结果")
    for img in tqdm(images, file=sys.stdout, ascii="░▒█", desc=" - 头部检测开始处理"):
        detected.append(detect_heads(img, level, max_infer_size, conf_threshold, iou_threshold))
    output_cache = detected
    return detected


def area_fill(dataset_name, is_random, color):
    global output_cache
    area = output_cache
    images = dataset_getImg(dataset_name)[0]
    fill = []
    # print(" - 区域填充开始处理")
    for img, xyxy in tzip(images, area, file=sys.stdout, ascii="░▒█", desc=" - 区域填充开始处理"):
        xyxy = xyxy[0]
        if is_random:
            color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        # print(img, eval(area), xyxy, xyxy[0][0])
        fill.append(censor_areas(img, 'color', [xyxy[0]], color=color))  # [xyxy[0][0]]
    # os.makedirs(f"processed/{dataset_name}", exist_ok=True)
    # for i, sv in enumerate(fill):
    #     sv.save(f"processed/{dataset_name}/{dataset_name}_AreaFill_{i+1}.png")
    output_cache = fill
    return fill


def crop_hw(dataset_name):
    global output_cache
    mask_info = output_cache
    images = dataset_getImg(dataset_name)[0]
    result = []
    for img, infos in zip(images, mask_info):
        infos = infos[0]
        (x0, y0, x1, y1) = infos[0]
        detect_type = infos[1]
        # score = infos[2]
        img_array = numpy.array(rgb_encode(img))
        h, w = img_array.shape[1:]
        mask = numpy.zeros((h, w))
        # print(mask, h, w)
        if detect_type == 'face' or detect_type == 'head':
            mask[y0:y1, x0:x1] = 1
            # print(mask)
        else:
            output_cache = []
            return "此内容不支持剪裁"
        result.append(squeeze(img, mask))
    output_cache = result
    return result


def crop_trans(dataset_name, threshold, filter_size):
    global output_cache
    images = dataset_getImg(dataset_name)[0]
    out = []
    # print(" - 自适应裁剪开始处理")
    for img in tqdm(images, file=sys.stdout, desc=" - 自适应裁剪开始处理", ascii="░▒█"):
        if img is not None:
            out.append(squeeze_with_transparency(img, threshold, filter_size))
    output_cache = out
    return out


def img_segment(dataset_name, scale):
    global output_cache
    images = dataset_getImg(dataset_name)[0]
    out = []
    # print(" - 人物分离开始处理")
    for img in tqdm(images, file=sys.stdout, desc=" - 人物分离开始处理", ascii="░▒█"):
        out.append(segment_rgba_with_isnetis(img, scale)[1])  # mask信息被丢弃了
    output_cache = out
    return out


def ref_datasets(need_list=False):
    # 遍历本地数据集
    list_datasets = []
    with os.scandir("dataset") as datasets:
        for each_dataset in datasets:
            # f_dataset = each_dataset.__next__()
            if not each_dataset.name.startswith('.') and each_dataset.is_dir():
                list_datasets.append(each_dataset.name)
    if need_list:
        return list_datasets
    else:
        return gr.Dropdown.update(choices=list_datasets)


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
        return gr.Dropdown.update(choices=custom_blacklist)


def ref_runs(dataset_name, need_list=False):
    runs_list = []
    try:
        with os.scandir(f"runs/{dataset_name}/ckpts") as conv_list:
            for conv in conv_list:
                # print("遍历了一个conv")
                if conv.is_file():
                    if conv.name.endswith('.pt'):
                        runs_list.append(conv.name.replace(dataset_name+"-", "").replace(".pt", ""))
    except FileNotFoundError:
        if need_list:
            return []
        else:
            return gr.Dropdown.update(choices=[])
    if runs_list is not None:
        runs_list = sorted(runs_list, key=int, reverse=True)
        if need_list:
            return runs_list
        else:
            # print("结果"+str(runs_list))
            return gr.Dropdown.update(choices=runs_list)


def convert_weights(dataset_name, step):
    global output_cache
    # logging.try_init_root(logging.INFO)
    convert_to_webui_lora(f"runs/{dataset_name}/ckpts/unet-{step}.safetensors",
                          f"runs/{dataset_name}/ckpts/text_encoder-{step}.safetensors",
                          os.path.join(f"runs/{dataset_name}/ckpts", f"{dataset_name}-lora-{step}.safetensors")
                          )
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
        os.makedirs(process_dir, exist_ok=True)
        anyfiles = os.listdir(process_dir)
        # print(" - 开始保存运行结果")
        for anyfile in anyfiles:
            os.remove(f"{process_dir}/{anyfile}")
        for i, sv in enumerate(tqdm(output_cache, file=sys.stdout, desc=" - 开始保存运行结果", ascii="░▒█")):
            sv.save(f"{process_dir}/{dataset_name}_{i+1}.png")
            count = count+1
        output_cache = []
        return "已保存 "+str(count)+" 张图片至"+process_dir+"文件夹"


def tagging_main(dataset_name, ttype, wd14_tagger, wd14_general_thre, wd14_character_thre, wd14_weight, wd14_overlap, ml_real_name, ml_thre, ml_scale, ml_weight, ml_ratio, ml_overlap, need_black, drop_presets, drop_custom, exists_txt, del_json):
    global output_cache
    images = dataset_getImg(dataset_name)[0]
    img_name = images[1]
    result = []
    if ttype == taggers[0]:
        # print(" - 数据打标开始处理")
        for img, name in tzip(images, img_name, file=sys.stdout, ascii="░▒█", desc=" - 数据打标开始处理"):
            result = get_wd14_tags(img, wd14_tagger, wd14_general_thre, wd14_character_thre, wd14_overlap)
            if result[2]:
                result = tags_to_text(result[1], include_score=wd14_weight)+', '+tags_to_text(result[2], include_score=wd14_weight)  # features and chars
            if need_black:
                result = str(str(drop_blacklisted_tags([result], drop_presets, drop_custom))[2:-2])
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
    elif ttype == taggers[1]:
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
    elif ttype == taggers[2]:
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
                output_cache = []
                return "无标签"
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


# 主界面
with gr.Blocks(css="style.css", analytics_enabled=False) as iblock:
    output_cache = []
    quicksettings = gr.Row(elem_id="quicksettings")
    with quicksettings:
        dataset_dropdown = gr.Dropdown(ref_datasets(True), label="当前数据集", value=ref_datasets(True)[0], container=True, show_label=True, interactive=True, elem_id='dataset_dropbar')
        ref_datasets_button = gr.Button("🔄", elem_id='refresh_datasets')
    with gr.Tab("数据集获取"):
        source = gr.Radio(['Danbooru', 'Pixiv', 'Zerochan'], label='选择图站', value='Danbooru')
        char_name = gr.Textbox(label='角色名称', value='', placeholder='填入角色名')
        pre_min_size = gr.Textbox(label="最小尺寸", value="600", interactive=False)
        pre_background = gr.ColorPicker(label="背景色", value="#FFFFFF", interactive=False)
        pre_class = gr.CheckboxGroup(["素描", "漫画", "3D"], label="风格过滤", value=None, type="index", interactive=False)
        pre_rating = gr.CheckboxGroup(["健全", "r15", "r18"], label="评级过滤", value=["健全"], type="index", interactive=False)
        pre_crop_person = gr.Checkbox(label="裁剪人物", value=False, interactive=False)
        pre_auto_tagging = gr.Checkbox(label="自动打标", value=False, interactive=False)
        with gr.Column(visible=False) as pixiv_settings:
            pixiv_no_ai = gr.Checkbox(label="非AI生成", interactive=True, value=False)
            pixiv_token = gr.Textbox(label="刷新令牌", placeholder="必须: pixiv的refresh token", interactive=True)
            pixiv_get_token = gr.Button("前往查询", interactive=True)
            with gr.Accordion("令牌说明", open=False):
                gr.Markdown("获取Pixiv图片需要刷新令牌\n"
                            "用法：点击`前往获取`，将打开Pixiv网页，按F12启用开发者控制台，选择`网络/Network`，点击左侧第三个按钮`筛选器`，"
                            "筛选`callback?`点击继续使用此账号登陆，此时页面会跳转，开发者控制台会出现一条请求，点击它，进入`标头`"
                            "复制`code=`后的内容，填入后台（黑窗口）按回车，后台将返回你的refresh token\n"
                            "取消获取请按ctrl+c")
        source.select(pixiv_setting_ctrl, None, [pixiv_settings])
        dl_count = gr.Slider(1, 100, label='下载数量', value=10, step=1)
        # save_path = gr.Textbox(label='保存路径', value='dataset', placeholder='自动创建子文件夹')
        download_button = gr.Button("获取图片", variant="primary", interactive=True)
        with gr.Accordion("使用说明", open=False):
            gr.Markdown("数据集详细说明..什么的")
        pre_rating.change(pre_rating_limit, [pre_rating], [download_button])
        pixiv_get_token.click(get_ref_token, [], [])
    with gr.Tab("数据集处理"):
        with gr.Accordion("三阶分割"):
            stage_button = gr.Button("开始处理", variant="primary")
        with gr.Accordion("差分过滤"):
            cluster_button = gr.Button("开始处理", variant="primary")
            with gr.Accordion("使用说明", open=False):
                gr.Markdown("差分检测：LPIPS（感知图像补丁相似性） ，全称为Learned Perceptual Image Patch "
                            "Similarity，是一种用于评估图像相似性的度量方法。基于深度学习模型，通过比较图像之间的深度特征评估它们的相似性\n "
                            "LPIPS使用了预训练的分类网络（如AlexNet或VGG）来提取图像的特征。然后计算两个图像特征之间的余弦距离，"
                            "并对所有层和空间维度的距离进行平均，可以得到一个值，用于表示两个图像之间的感知差异。\n"
                            "*会返回去除差分后的图片结果"
                            "![cluster](file/markdown_res/lpips_full.plot.py.svg)")
        with gr.Accordion("人物分离"):
            seg_scale = gr.Slider(32, 2048,label="缩放大小", info="图像传递给模型时的缩放尺寸", step=32, value=1024, interactive=True)
            with gr.Accordion("使用说明", open=False):
                gr.Markdown("人物分离\n"
                            "*会返回背景为透明的人物图片结果\n"
                            "查阅skytnt的[复杂动漫抠像](https://github.com/SkyTNT/anime-segmentation/)")
            seg_button = gr.Button("开始处理", variant="primary")
        # with gr.Accordion("人物检测"):
        #     ccip_level = gr.Checkbox(label="使用高精度", value=True, interactive=True)
        #     ccip_model = gr.Dropdown(["v0", "v1", "v1.1"], label="模型选择", value="v1.1", interactive=True)
        #     ccip_infer = gr.Slider(32, 2048, label="缩放大小", interactive=True, step=32, value=640, info="图像传递给模型时的缩放尺寸")
        #     ccip_conf = gr.Slider(0.01, 1, label="检测阈值", interactive=True, value=0.25, step=0.01, info="置信度高于此值的检测结果会被返回")
        #     ccip_iou = gr.Slider(0.01, 1, label="重叠阈值", interactive=True, value=0.7, step=0.01, info="重叠区域高于此阈值将会被丢弃")
        #     ccip_button = gr.Button("开始检测", variant="primary")
        #     with gr.Accordion("使用说明", open=False):
        #         gr.Markdown("角色检测：CCIP（对比角色图像预训练）从动漫角色图像中提取特征，计算两个角色之间的视觉差异，并确定两个图像是否"
        #                     "描绘相同的角色。![ccip](file/markdown_res/ccip_full.plot.py.svg)"
        #                     "更多信息可查阅 [CCIP官方文档](https://deepghs.github.io/imgutils/main/api_doc/metrics/ccip.html).")
        with gr.Accordion("面部检测"):
            faced_level = gr.Checkbox(value=True, label="使用高精度", interactive=True)
            faced_model = gr.Dropdown(["v0", "v1", "v1.3", "v1.4"], label="模型选择", value="v1.4", interactive=True)
            faced_infer = gr.Slider(32,2048, label="缩放大小", interactive=True, step=32, value=640, info="图像传递给模型时的缩放尺寸")
            faced_conf = gr.Slider(0.01, 1, label="检测阈值", interactive=True, value=0.25, step= 0.01, info="置信度高于此值的检测结果会被返回")
            faced_iou = gr.Slider(0.01, 1, label="重叠阈值", interactive=True, value=0.7, step=0.01, info="重叠区域高于此阈值将会被丢弃")
            with gr.Accordion("使用说明", open=False):
                gr.Markdown("##面部检测"
                            "来自imgutils检测模块"
                            "###此功能会返回一个区域结果，而不是图片结果")
            faced_button = gr.Button("开始检测", variant="primary")
        with gr.Accordion("头部检测"):
            headd_level = gr.Checkbox(value=True, label="使用高精度", interactive=True)
            headd_infer = gr.Slider(32,2048, label="缩放大小", interactive=True, step=32, value=640, info="图像传递给模型时的缩放尺寸")
            headd_conf = gr.Slider(0.01, 1, label="检测阈值", interactive=True, value=0.25, step=0.01, info="置信度高于此值的检测结果会被返回")
            headd_iou = gr.Slider(0.01, 1, label="重叠阈值", interactive=True, value=0.7, step=0.01, info="重叠区域高于此阈值将会被丢弃")
            with gr.Accordion("使用说明", open=False):
                gr.Markdown("##头部检测"
                            "来自imgutils检测模块"
                            "###此功能会返回一个区域结果，而不是图片结果")
            headd_button = gr.Button("开始检测", variant="primary")
        with gr.Accordion("区域填充"):
            areaf_isRandom = gr.Checkbox(label="随机颜色", value=True, interactive=True)
            areaf_color = gr.ColorPicker(label="自定义颜色", value="#00FF00", visible=not areaf_isRandom.value)
            areaf_button = gr.Button("开始处理", variant="primary")
            with gr.Accordion("使用说明", open=False):
                gr.Markdown("接收输出后的结果进行打码。\n"
                            "运行结果内有区域信息，才可以填充...")
            areaf_isRandom.select(color_picker_ctrl, None, [areaf_color])
        with gr.Accordion("区域剪裁"):
            crop_hw_button = gr.Button("开始处理", variant="primary")
            with gr.Accordion("使用说明", open=False):
                gr.Markdown("将运行结果中的区域进行剪裁。\n"
                            "运行结果内有区域信息，才可以剪裁...")
        with gr.Accordion("自适应剪裁"):
            crop_trans_button = gr.Button("开始处理", variant="primary")
            crop_trans_thre = gr.Slider(0.01, 1, label="容差阈值", value=0.7, step=0.01)
            crop_trans_filter = gr.Slider(0, 10, label="羽化", value=5, step=1)
            with gr.Accordion("使用说明", open=False):
                gr.Markdown("将数据集中的透明图片进行自适应剪裁。\n"
                            "不对运行结果中的内容进行操作。")
    with gr.Tab("打标器"):
        taggers = ["wd14", "mldanbooru", "json解析"]
        tagger_type = gr.Dropdown(taggers, value=taggers[0], label="打标器", allow_custom_value=False, interactive=True)
        with gr.Column(visible=tagger_type.value == taggers[0]) as tagger_wd14_settings:
            wd14_tagger_model = gr.Dropdown(["SwinV2", "ConvNext", "ConvNextV2", "ViT", "MOAT"], value="ConvNextV2", label="打标模型", interactive=True)
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
                gr.Markdown("用此脚本获取的图片附有json文件\n"
                            "使用此打标器以从中提取tag\n"
                            "此功能不会检查图片，而是从所有可能的json文件中提取tag")
            anal_del_json = gr.Checkbox(value=False, label="删除json", interactive=True)
        use_blacklist = gr.Checkbox(label="使用黑名单", value=True, interactive=True)
        with gr.Column(visible=use_blacklist.value) as tagger_dropper_settings:
            drop_use_presets = gr.Checkbox(value=True, label="使用在线黑名单", info="获取在线黑名单，来自alea31435", interactive=True)
            with gr.Column(visible=not drop_use_presets.value, elem_id="drop_custom_setting") as drop_custom_setting:
                drop_custom_list = gr.Dropdown(ref_customList(True), value=ref_customList(True)[0], label="自定义黑名单", elem_id="custom_list", interactive=True, info="黑名单路径cfgs/blacklist/")
                drop_ref_button = gr.Button("🔄", elem_id='refresh_custom_list')
        op_exists_txt = gr.Dropdown(["复制文件", "忽略文件", "覆盖文件", "附加标签"], value="附加标签", info="对于已存在标签，打标器的行为", show_label=False, interactive=True)
        tagger_button = gr.Button("打标", variant="primary")
        # tagger_type.select(tagger_chooser_ctrl, None, [globals()[f'tagger_{("dropper" if tagger == "标签黑名单" else tagger)}_settings'] for tagger in taggers])
        tagger_type.select(tagger_chooser_ctrl, None, [globals()[f'tagger_{("anal" if tagger == "json解析" else tagger)}_settings'] for tagger in taggers])
        # wd14_use_blacklist.select(blacklist_settings_ctrl, None, [tagger_dropper_settings])
        # ml_use_blacklist.select(blacklist_settings_ctrl, None, [tagger_dropper_settings])
        use_blacklist.select(blacklist_settings_ctrl, None, [tagger_dropper_settings])
        drop_use_presets.select(custom_blacklist_ctrl, None, [drop_custom_setting])
    with gr.Tab("PLoRA训练"):
        min_step = gr.Textbox(label="最小步数", value='', placeholder='不填写将自动计算')
        train_button = gr.Button("开始训练", variant="primary")
        with gr.Accordion("权重合并", open=True):
            with gr.Column(elem_id="convert_lora_steps") as convert_lora_steps:
                convert_step = gr.Dropdown(ref_runs(dataset_dropdown.value, True), value=ref_runs(dataset_dropdown.value, True)[0] if ref_runs(dataset_dropdown.value, True) else [], label="步数", info="合并对应步数的权重文件", elem_id="convert_list", multiselect=False, interactive=True)
                convert_ref_button = gr.Button("🔄", elem_id='convert_ref_button')
            convert_weights_button = gr.Button("开始合并", variant="primary")
        with gr.Accordion("使用说明", open=False):
            gr.Markdown("训练详细说明..什么的")
    with gr.Tab("质量验证"):
        with gr.Accordion("使用说明", open=False):
            gr.Markdown("soon...")
    with gr.Column(elem_id="output"):
        message_output = gr.Textbox(label='运行结果', elem_id="message_output")
        save_output = gr.Button("💾", elem_id="save_output", interactive=False)
        message_output.change(save_output_ctrl, [], save_output)
    download_button.click(download_images, [source, char_name, pre_min_size, pre_background, pre_class, pre_rating, pre_crop_person, pre_auto_tagging, dl_count, pixiv_token, pixiv_no_ai], [message_output], scroll_to_output=True)
    ref_datasets_button.click(ref_datasets, [], [dataset_dropdown])
    stage_button.click(three_stage, [dataset_dropdown], [message_output])
    drop_ref_button.click(ref_customList, [], [drop_custom_list])
    convert_ref_button.click(ref_runs, [dataset_dropdown], [convert_step])
    convert_weights_button.click(convert_weights, [dataset_dropdown, convert_step], [message_output])
    cluster_button.click(clustering, [dataset_dropdown], [message_output], scroll_to_output=True)
    seg_button.click(img_segment, [dataset_dropdown, seg_scale], [message_output], scroll_to_output=True)
    # ccip_button.click(person_detect, [dataset_dropdown, ccip_level, ccip_model, ccip_infer, ccip_conf, ccip_iou], [message_output])
    faced_button.click(face_detect, [dataset_dropdown, faced_level, faced_model, faced_infer, faced_conf, faced_iou], [message_output], scroll_to_output=True)
    headd_button.click(head_detect, [dataset_dropdown, headd_level, headd_infer, headd_conf, headd_iou], [message_output], scroll_to_output=True)
    train_button.click(run_train_plora, [dataset_dropdown, dataset_dropdown, min_step], [message_output], scroll_to_output=True)
    areaf_button.click(area_fill, [dataset_dropdown, areaf_isRandom, areaf_color], [message_output], scroll_to_output=True)
    crop_hw_button.click(crop_hw, [dataset_dropdown], [message_output], scroll_to_output=True)
    crop_trans_button.click(crop_trans, [dataset_dropdown, crop_trans_thre, crop_trans_filter], [message_output], scroll_to_output=True)
    tagger_button.click(tagging_main, [dataset_dropdown, tagger_type, wd14_tagger_model, wd14_general_threshold, wd14_character_threshold, wd14_format_weight, wd14_drop_overlap, ml_use_real_name, ml_threshold, ml_size, ml_format_weight, ml_keep_ratio, ml_drop_overlap, use_blacklist, drop_use_presets, drop_custom_list, op_exists_txt, anal_del_json], [message_output], scroll_to_output=True)
    save_output.click(saving_output, [dataset_dropdown], [message_output])
    iblock.title = "小苹果webui"

if __name__ == "__main__":
    iblock.launch()
