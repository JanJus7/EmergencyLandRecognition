import os

aircrafts = ["c152", "b737"]
images = ["autobahn", "bay", "bva", "gdn", "kromeriz", "smol"]

for ac in aircrafts:
    for img in images:
        print(f"\n---> Aircraft: {ac}, Pic: {img}")
        os.system(f"python predict.py -a {ac} -i {img}")
        
print("\nDONE!")