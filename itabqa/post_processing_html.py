import os

import pandas as pd
from bs4 import BeautifulSoup, Comment


def post_process_table_html(html_table):
    # detect hierarchical
    hierarchical_flag = False
    hierarchical_td_store = ''

    table_ = BeautifulSoup(html_table, "lxml")
    tr_list = table_.find_all('tr')
    for tr in tr_list:
        th_td_list = tr.find_all('td') + tr.find_all('th')
        # check the first <td> in each row
        if (th_td_list[0].has_attr('colspan') and not th_td_list[0].has_attr('rowspan')) \
                or (len(th_td_list[1].contents) == 0):
            hierarchical_flag = True
            hierarchical_td_store = th_td_list[0].contents[0]
            continue

        # if the first td is rowspan, then reset hierarchical_flag
        if th_td_list[0].has_attr('rowspan'):
            hierarchical_flag = False
            hierarchical_td_store = ''
            continue

        if hierarchical_flag:
            tmp = hierarchical_td_store + ',,,' + th_td_list[0].contents[0]
            th_td_list[0].string = tmp

    return table_.prettify(formatter=None)#table_.find_all('table')[0]


infer_html_path = '/disks/strg16-176/nam/VQAonBD2023/debug_hierar/'

for file_name in os.listdir(infer_html_path):
    f_path = os.path.join(infer_html_path, file_name)
    if file_name.endswith('.txt'):
        with open(f_path) as f:
            html_ = f.read()

        html_ = html_.replace('\n', '')
        html_ = post_process_table_html(html_)
        table_pd = pd.read_html(html_)
        print(table_pd)
