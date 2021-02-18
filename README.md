# RobotSim_Yolov4
YOLOv4 in Robot Simulation environment

Clone this repo
git clone https://github.com/yunsujeon/RobotSim_Yolov4.git

이 레포지토리는 두개의 레포지토리 환경을 참고했다.

RobotSim_Yolo4 레포 들어가서 두개 더 클론해온다. 
habitat-sim
yolov4

install 해야됨


script를 사용해 비디오를 녹화하고 불러옴 
Python dev1.py -simvid 1 -record 1 //yolo와 sim기능
python dev1.py -simvid 1 // 있는 비디오로 추론만 

이미지 파일 yolo
python dev1.py -imgfile <filepath>

웹캠 yolo
python dev1.py -webcam 1

화면의 일정 영역 crop하여 yolo
python dev1.py -crop 1

시뮬레이션 API를 이용한 실시간 yolo 
개발중
(python dev1.py -simframe 1)

