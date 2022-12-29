data = []
with open ("origin_video.txt", "r") as f:
    f = f.read()
    f = f.split("\n")

for i in f:
    i = i.split(" ")
    data.append(i)

for frame in data:
    print(frame[1])
       
# print(f)