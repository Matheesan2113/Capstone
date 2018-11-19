import json
import glob

#result = []
#for f in glob.glob("*.json"):
#    with open(f, "rb") as infile:
#        result.append(json.load(infile))

#with open("merged_file.json", "wb") as outfile:
#     json.dump(result, outfile)


#read_files = glob.glob("*.json")
#with open("merged_file.json", "wb") as outfile:
#    outfile.write('[{}]'.format(
#        b','.join(
#                [open(f, "rb")
#                .read() for f in read_files]
#               )))
read_files = glob.glob("*.json")
output_list = []

for f in read_files:
    with open(f, "r") as infile:
        output_list.append(json.load(infile))

with open("merged_file.json", "wb") as outfile:
    json.dump(output_list, outfile)