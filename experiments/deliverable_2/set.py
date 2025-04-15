import os 
cmd="python /home/vpa/RAGollama3/experiments/deliverable_2/6.dialouge_gen_api_rectify.py --idx {} &"

for i in range(0,35):
    os.system(cmd.format(i))

# for i in range(0,5):
#     os.system(cmd.format(i))