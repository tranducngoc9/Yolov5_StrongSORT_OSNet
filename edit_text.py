
with open ("runs/track/exp/tracks/2022-10-05-09-30_cut.txt", "r") as f:
    f = f.read()
    f = f.split("\n")
    
for line in f:
    line_split = line.split(" ")
     
    if line_split[1] == "123":
        print(line)
        f.remove(line)

for line in f:
    with open("filter_text.txt","a+") as file:
        file.write(line + "\n")


