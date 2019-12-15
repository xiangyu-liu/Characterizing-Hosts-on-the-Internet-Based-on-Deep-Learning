from CitationRank import TargetContents
import json
import socket
import pickle
import multiprocessing
from multiprocessing.dummy import Pool

log_path = "/newNAS/Workspaces/DRLGroup/xiangyuliu/Misc/logs/"
ip_path = "/newNAS/Workspaces/DRLGroup/xiangyuliu/Misc/ip/"
black_list_path = [r"/newNAS/Workspaces/DRLGroup/xiangyuliu/Misc/domain/wrz.txt",
                   r"/newNAS/Workspaces/DRLGroup/xiangyuliu/Misc/domain/ad_servers.txt",
                   r"/newNAS/Workspaces/DRLGroup/xiangyuliu/Misc/domain/emd.txt",
                   r"/newNAS/Workspaces/DRLGroup/xiangyuliu/Misc/domain/exp.txt",
                   r"/newNAS/Workspaces/DRLGroup/xiangyuliu/Misc/domain/fsa.txt",
                   r"/newNAS/Workspaces/DRLGroup/xiangyuliu/Misc/domain/grm.txt",
                   r"/newNAS/Workspaces/DRLGroup/xiangyuliu/Misc/domain/hfs.txt",
                   r"/newNAS/Workspaces/DRLGroup/xiangyuliu/Misc/domain/hjk.txt",
                   r"/newNAS/Workspaces/DRLGroup/xiangyuliu/Misc/domain/mmt.txt",
                   r"/newNAS/Workspaces/DRLGroup/xiangyuliu/Misc/domain/pha.txt",
                   r"/newNAS/Workspaces/DRLGroup/xiangyuliu/Misc/domain/psh.txt",
                   r"/newNAS/Workspaces/DRLGroup/xiangyuliu/Misc/domain/psh.txt",
                   r"/newNAS/Workspaces/DRLGroup/xiangyuliu/Misc/domain/psh.txt",
                   r"/newNAS/Workspaces/DRLGroup/xiangyuliu/Misc/domain/verified_online.json"]


def domain2ip(domain):
    result = socket.gethostbyname(domain)
    return result

num = 0


def get_black_ip_set(file):
    print(file)
    num = 0

    def process(domain):
        global num
        try:
            ip = domain2ip(domain)
            num += 1
            return ip
        except:
            print("ip error", domain, num)
            return "0"

    if "json" in file:
        json_dict = json.load(open(file))
        domain_list = [per_dict["url"] for per_dict in json_dict]
    else:
        with open(file, mode="r") as black_list:
            lines = black_list.readlines()[10: -1]
            domain_list = [line.split("\t")[1].split("\n")[0] for line in lines]

    cores = multiprocessing.cpu_count()
    print(cores)
    pool = Pool(processes=cores)
    results = pool.map(process, domain_list)
    pool.close()
    pool.join()
    return list(set(results))


file_count = 0


def main(black_ip_list):
    def fetch_json(ip):
        global file_count
        file_count += 1
        url = "https://censys.io/ipv4/" + ip + "/raw"
        print(file_count, ip)

        def fetch_dict():
            try:
                contents = TargetContents(url=url)
                return contents.soup.find_all("code")
            except:
                print("error")
                return []

        while True:
            parsed_html = fetch_dict()
            if len(parsed_html) != 0:
                break
            else:
                print("looping")

        for parsed_text in parsed_html:
            ip_dict = json.loads(parsed_text.get_text())
            json.dump(ip_dict, open(log_path + ip + ".json", mode="w"))

    cores = multiprocessing.cpu_count()
    print(cores)
    pool = Pool(processes=cores)
    pool.map(fetch_json, black_ip_list)
    pool.close()
    pool.join()


def run_scrapy():
    new_list = []
    for i in range(4):
        new_list = new_list + pickle.load(open(ip_path + str(i) + ".pkl", mode="rb"))
    new_list = list(set(new_list))
    pickle.dump(new_list, open(ip_path + "0123.pkl", mode="wb+"))
    main(new_list)


def run_domain2ip():
    i = -1
    for file_path in black_list_path:
        i += 1
        if i<= 3:
            continue
        ip_list = get_black_ip_set(file_path)
        pickle.dump(ip_list, open(ip_path + str(i) + ".pkl", mode='wb+'))
        print(len(ip_list), file_path, "finish:" + str(i))

if __name__ == '__main__':
    run_scrapy()
