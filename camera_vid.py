import imp
import os
import cv2
import argparse

def get_args():
    parser = argparse.ArgumentParser(description="Drinks object detection")
    parser.add_argument("--index", default=2, type=int, metavar="N")    
    return parser

def main(args):
    cap = cv2.VideoCapture(args.index)
    fourcc = cv2.VideoWriter_fourcc(*"MP4V")
    vidpath = "./demo/input.mp4"

    if not os.path.exists("./demo"):
        os.mkdir("./demo")

    out = cv2.VideoWriter(vidpath,fourcc, 10.0, (640,480))
    
    while(True):
        print("Recording... ")
        ret, frame = cap.read()
        out.write(frame)
        print(ret)
        if ret:
            cv2.imshow("demo",frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    print("Done!")
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    cv2.waitKey(1)

if __name__ == "__main__":
    args = get_args().parse_args()
    main(args)