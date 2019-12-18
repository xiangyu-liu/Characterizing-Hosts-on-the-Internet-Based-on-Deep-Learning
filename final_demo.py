import argparse
import json
from ScrapyCensys import domain2ip
from ScrapyCensys import fetch_json
from app1_demo.XGB_highdim import demo1_highdim
from app1_demo.XGB_lowdim import demo1_lowdim
from app2_high.demo3 import demo2_high
from app2_low.demo import demo2_low


def main(args):
    if args.ip == None:
        try:
            ip = domain2ip(args.url)
        except:
            print("url error")
            return
    else:
        ip = args.ip
    print("url is {} ip address is {}".format(args.url, ip))

    if args.load_json:
        content_dict = json.load(open(r"C:\Users\11818\Desktop\RL\Code\vae\data.json"))
    else:
        if args.search_input == "ip":
            content_dict = fetch_json(ip, store=False, url_direct=False)
        else:
            content_dict = fetch_json(args.url, store=False, url_direct=True)

    if "demo1" in args.demo:
        print("###begin test high embedding###")
        demo1_highdim(content_dict)
        print("###begin to test low embedding###")
        demo1_lowdim(content_dict)
    elif "demo2" in args.demo:
        print("###begin test high embedding###")
        demo2_high(content_dict)
        print("###begin to test low embedding###")
        demo2_low(content_dict)
    else:
        print("no such demo")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='demo')
    parser.add_argument("--search_input", default="url", type=str)
    parser.add_argument("--url", default="google.com", type=str)
    parser.add_argument("--ip", default=None, type=str)
    parser.add_argument("--demo", default="demo2", type=str)
    parser.add_argument("--load_json", default=False, action="store_true")
    args = parser.parse_args()
    main(args)
