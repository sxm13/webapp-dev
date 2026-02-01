import time
import requests


FILE_URL = "https://raw.githubusercontent.com/fxcoudert/MOF_papers/master/posted.dat"

LOCAL_FILE = "./data/mofpaper/posted_local.dat"

def fetch_file_content(url):
    response = requests.get(url)
    if response.status_code == 200:
        return response.text.splitlines()
    else:
        raise Exception(f"Failed to fetch file: {response.status_code}")

def compare_files(local_file, remote_lines):
    try:
        with open(local_file, "r") as f:
            local_lines = f.read().splitlines()
    except FileNotFoundError:
        local_lines = []

    new_lines = [line for line in remote_lines if line not in local_lines]

    with open(local_file, "w") as f:
        f.write("\n".join(remote_lines))

    return new_lines

slack_webhook_url = "https://hooks.slack.com/services/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"

def ElsevierAPI(url, api_key):
    url = url
    headers = {
        'X-ELS-APIKey': api_key,
        'Accept': 'application/json'  
    }
    response = requests.get(url, headers=headers)
    data = response.json()
    doi = data.get('full-text-retrieval-response', {}).get('coredata', {}).get('prism:doi', 'DOI not found')
    return doi

def run():

    Elsevier_api_key = '3e15b3e71f7cd8229aa275645c62256f'

    while True:
        try:
            print("Checking for updates...")
            remote_lines = fetch_file_content(FILE_URL)
            
            new_lines = compare_files(LOCAL_FILE, remote_lines)
            
            if new_lines:

                for link in new_lines:
                    
                    if len(link.split(":"))>1:
                        try:
                            # RSC
                            if link.split("//")[1].split("/")[0] == "pubs.rsc.org":
                                DOI = "10.1039/" + link.split("/")[-1]
                               
                            # ACS
                            elif link.split("//")[1].split("/")[1] == "10.1021":
                                DOI = link.replace("http://dx.doi.org/","")
                               
                            # Nature
                            elif link.split("//")[1].split("/")[0] == "www.nature.com":
                                DOI = "10.1038/" + link.split("/")[-1]
                               
                            # Science
                            elif link.split("//")[1].split("/")[0] == "www.science.org":
                                DOI = "10.1126/" + link.split("/")[-1]
            
                            # Wiley
                            elif link.split("//")[1].split("/")[0] == "onlinelibrary.wiley.com" or link.split("//")[1].split("/")[0] == "pericles.pericles-prod.literatumonline.com":
                                DOI = "10.1002/" + link.split("/")[-1].split("?")[0]
                             
                            # AIP
                            elif link.split("//")[1].split("/")[0] == "aip.scitation.org":
                                DOI = "10.1063/" + link.split("/")[-1].split("?")[0]
                                
                            # Elsevier
                            elif link.split("//")[1].split("/")[0] == "www.sciencedirect.com":
                                DOI = ElsevierAPI(link,Elsevier_api_key)
                                check = False
                            # Cell
                            elif link.split("//")[1].split("/")[0] == "www.cell.com":
                                link_ = "https://api.elsevier.com/content/article/pii/" + link.split("/")[-1].split("?")[0]
                                DOI = ElsevierAPI(link_,Elsevier_api_key)
                                
                            else:
                                print(link, "wait for checking")
                        except Exception as e:
                            print(f"{e}")
                    else:
                        DOI = link
                        
                    return DOI

            else:   
                pass
        except Exception as e:
            print(f"{e}")

        time.sleep(3600)


# slack_payload = {
#                 "text": (
#                             f"new MOF paper has posted by @MOF_paper:\n"
#                             f"paper link: {new_lines}\n"
#                         )
#                 }
# response = requests.post(slack_webhook_url, json=slack_payload)
# print(f"New MOF Paper:\n{new_lines}")


