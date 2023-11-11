import os.path

import requests
import re
from bs4 import BeautifulSoup


def get_image_links(user_id):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/93.0.4577.82 Safari/537.36 Edg/93.0.961.52",
    }
    res = requests.get(f"https://www.pixiv.net/ajax/user/{user_id}/profile/all?lang=zh", headers=headers)
    if res.status_code == 200:
        p = res.text
        user_illust = re.findall(r'(?<=\")\d+(?=\":null)', p)
        url_num_split = -(-len(user_illust) // 100)
        new_url = []
        for i in range(url_num_split):
            new_url.append(f"https://www.pixiv.net/ajax/user/{user_id}/profile/illusts?")
            max_val = min((i+1)*100, len(user_illust))
            for j in range(i*100, max_val):
                new_url[i] += f"ids%5B%5D={user_illust[j]}&"
            new_url[i] += "work_category=illustManga&is_first_page=0&lang=zh"
        text_final = []
        pic_name = []
        for url in new_url:
            res = requests.get(url, headers=headers)
            if res.status_code == 200:
                p = res.text
                illust_id_page = re.findall(r'\d+\*\*\d+', re.sub(r'","title":".*?"pageCount":', '**', p))
                for k in illust_id_page:
                    if k.split("**")[1] == 1:
                        text_final.append(f"https://pixiv.re/{k.split('**')[0]}.png")
                        pic_name.append(f"{k.split('**')[0]}")
                    else:
                        for h in range(int(k.split("**")[1])):
                            text_final.append(f"https://pixiv.re/{k.split('**')[0]}-{h+1}.png")
                            pic_name.append(f"{k.split('**')[0]}-{h+1}")
        return text_final, pic_name


# def get_user_id(artist_name, cookies):
#     headers = {
#         "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/93.0.4577.82 Safari/537.36 Edg/93.0.961.52",
#     }
#     res = requests.get(f"https://www.pixiv.net/search_user.php?s_mode=s_usr&nick={artist_name}", headers=headers, cookies=cookies)
#     if res.status_code == 200:
#         soup = BeautifulSoup(res.text, 'html.parser')
#         user_link = soup.find('a', title=artist_name)
#         if user_link:
#             user_id = user_link['href'].split('/')[-1]
#             return user_id
#     return None


def download_link(url, filename):
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(filename, 'wb') as file:
            for chunk in response.iter_content(1024):
                file.write(chunk)
