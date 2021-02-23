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

#### Install Requirements

1. Clone this repo

<pre><code>git clone https://github.com/yunsujeon/RobotSim_Yolov4.git</code></pre>

2. anaconda setup and activate

<pre><code>conda create -n name python==3.6.1</code></pre>

<pre><code>conda activate name</code></pre>

3. Install packages

<pre><code>cd RobotSim_Yolov4</code></pre>

<pre><code>pip install -r requirements.txt</code></pre>

  * nvidia graphic driver must be installed on your environment

4. Clone Habitat-Sim & Yolov4

<pre><code>git clone https://github.com/facebookresearch/habitat-sim.git</code></pre>

<pre><code>git clone https://github.com/Tianxiaomo/pytorch-YOLOv4.git</code></pre> 

5. Installation and download

  * Habitat-sim conda install
  
    * <pre><code>conda install habitat-sim -c conda-forge -c aihabitat</code></pre>
    
  * Yolov4 Weights Download
  
    * yolov4.pth(https://drive.google.com/open?id=1wv_LiFeCRYwtpkqREPeI13-gPELBDwuJ)
    
    * yolov4.conv.137.pth(https://drive.google.com/open?id=1fcbR0bWzYfIEdLJPzOsn4R5mlvR6IQyA)
    
  * locate all weight files on this path
  
    * RobotSim_Yolov4/pytorch-YOLOv4/
  
6. Build Habitat-Sim

<pre><code>cd habitat-sim</code></pre>

<pre><code>./build.sh</code></pre>

  * If you got a build error like "raise CalledProcessError ..etc.." , try again ./build.sh
  
* **If you got a common-testing-issues GL**

  * go to [link](https://github.com/facebookresearch/habitat-sim#common-testing-issues)
  
7. Download example map file / unzip correct location

  * habitat-test-scenes [link](https://drive.google.com/file/d/119Arq6EC-Jiz7gCFP3X49h1xtiijb2aa/view?usp=drivesdk)
  
    * locate RobotSim_Yolov4/
    
    * unzip habitat-test-scenes.zip
    
  * object download [link](https://drive.google.com/file/d/10yvIzSlQWMNRl9-iccTDOgPCT375mmOZ/view?usp=drivesdk)
  
    * locate RobotSim_Yolov4/data/objects/ (does not exist path -> mkdir)
    
    * unzip objects_v0.2.zip
    
  * mp3d download [link](https://drive.google.com/file/d/113yzi48RjyKOnoFwA-1WjaELiopuWSBd/view?usp=drivesdk)
  
    * locate RobotSim_Yolov4/data/scene_datasets/mp3d/ (does not exist path -> mkdir)
    
    * unzip mp3d_example.zip
    
  * In this repo's example code, use mp3d example map. You can use other maps with [this link](https://github.com/facebookresearch/habitat-sim#datasets)
  
#### Run Demo

1. Replace python files

dev1.py ---> RobotSim_Yolov4/pytorch-YOLOv4/

interaction.py ---> RobotSim_Yolov4/habitat-sim/

utils.py ---> RobotSim_Yolov4/pytorch-YOLOv4/tool/

AbstarctXApplication.cpp ---> RobotSim_Yolov4/habitat-sim/src/deps/magnum/src/Magnum/Platform/

RobotSim_Yolov4_1/build/viewer(Link to shared Library) -> RobotSim_Yolov4/pytorch-YOLOv4/

2. run dev1.py

* before run dev1.py, setting your path on interaction.py

  * line 35 / line 110 / line 601
  
<pre><code>cd pytorch-YOLOv4</code></pre> 

  * Record new script-made video and make yolo video
  
  <pre><code>python dev1.py -simvid 1 -record 1</code></pre>
  
    * check video > path is **habitat-sim/output/fetch.mp4**
    
  *  make yolo video with exist script-made video
  
  <pre><code>python dev1.py -simvid 1</code></pre>
  
  * API crop and yolo
  
  <pre><code>python dev1.py -crop 1</code></pre>

    * You have to move C++ API display to yolo's recognition region.

#### Make scripts your self

1. You can watch tutorial provided by facebook research [link](https://aihabitat.org/docs/habitat-sim/index.html)

2. In this tutorial, script codes are in interaction.py

  * Change or add python code, you can create new video.
  
#### C++ API change map

1. You can change map on C++ API

  * In this repo, you can run C++ API and yolo **python dev1.py -crop 1**
  
  * in dev1.py - def sub(): there exist a subprocess line
  <pre><code>subprocess.run(["./viewer", "--enable-physics", "../habitat-sim/data/scene_datasets/habitat-test-scenes/van-gogh-room.glb"])</code></pre>
  
  * Change code like this, you can run apartment_1.glb map file
  <pre><code>subprocess.run(["./viewer", "--enable-physics", "../habitat-sim/data/scene_datasets/habitat-test-scenes/apartment_1.glb"])</code></pre>


## Improvement

  * You can test&train with Many Map Datasets and Annotations [habitat-sim-datasets](https://github.com/facebookresearch/habitat-sim#datasets)




