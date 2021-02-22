# RobotSim_Yolov4
Robot Simulation (Habitat-Sim) & YOLOv4 python API connection

## Introduce

  * 로봇시뮬레이션 내에서 object detection을 실행함으로써 실환경 test 시간 비용 아끼는 효과
  * 사진, gif 등으로 소개
  * map DB중 annotation set 있다면 training 도 시키는 방향으로의 발전가능성


## Required Software

1. Habitat-Sim
https://github.com/Tianxiaomo/pytorch-YOLOv4
2. YOLOv4
https://github.com/facebookresearch/habitat-sim


## Run

### Install Requirements

1. anaconda setup and activate
<pre><code>conda create -n <name> python==3.6</code></pre>
<pre><code>conda activation <name></code></pre>

2. Install packages
pip install -r requirements.txt

3. Installation and download
  * Habitat-sim conda install
    * <pre><code>conda install habitat-sim -c conda-forge -c aihabitat</code></pre>
  * Yolov4 Weights Download
    * yolov4.pth(https://drive.google.com/open?id=1wv_LiFeCRYwtpkqREPeI13-gPELBDwuJ)
    * yolov4.conv.137.pth(https://drive.google.com/open?id=1fcbR0bWzYfIEdLJPzOsn4R5mlvR6IQyA)

### Run

1. Clone this repo
<pre><code>git clone https://github.com/yunsujeon/RobotSim_Yolov4.git</code></pre>

2. Clone Habitat-Sim & Yolov4   
<pre><code>cd RobotSim_Yolov4</code></pre>   
<pre><code>git clone https://github.com/facebookresearch/habitat-sim.git</code></pre>   
<pre><code>git clone https://github.com/Tianxiaomo/pytorch-YOLOv4.git</code></pre> 

3. Build Habitat-Sim
<pre><code>cd habitat-sim</code></pre>  
<pre><code>./build.sh</code></pre>  

4. Relocation python files
dev1.py -> RobotSim_Yolov4/pytorch-YOLOv4
interaction.py -> RobotSim_Yolov4/habitat-sim
utils.py -> 

5. run dev1.py
<pre><code>cd pytorch-YOLOv4</code></pre> 
  * Record new script-made video and make yolo video
  <pre><code>Python dev1.py -simvid 1 -record 1</code></pre> 
  *  make yolo video with exist script-made video
  <pre><code>python dev1.py -simvid 1</code></pre>
  * API crop and yolo
  <pre><code>  python dev1.py -crop 1</code></pre>


## Improvement

  * You can test&train with Many Map Datasets and Annotations [habitat-sim-datasets](https://github.com/facebookresearch/habitat-sim#datasets)



