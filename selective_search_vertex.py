#!/usr/bin/env python
'''
Installation:
pip uninstall opencv-contrib-python opencv-python
pip install opencv-contrib-python

Usage:
    python selective_search.py -p test.jpg -s
    python selective_search.py -d images/train -ft jpg,png -t f
    python selective_search.py -d images/train -t q -o test.pkl -i train.txt 
Use "l" to display less rects, 'm' to display more rects, "q" to quit.
'''
 
import os
import cv2
import argparse
from tqdm import tqdm
import pickle
import numpy as np

def selective_search(img_path, ss_type='q'):
    '''selective search result generation'''

    # read image
    im = cv2.imread(img_path)
    # resize image
    newHeight = 400
    newWidth = int(im.shape[1] * newHeight / im.shape[0])
    im = cv2.resize(im, (newWidth, newHeight))
    
    # create Selective Search Segmentation Object using default parameters
    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
 
    # set input image on which we will run segmentation
    ss.setBaseImage(im)
    
    # Switch to fast but low recall Selective Search method
    if (ss_type == 'f'):
        ss.switchToSelectiveSearchFast()
 
    # Switch to high recall but slow Selective Search method
    elif (ss_type == 'q'):
        ss.switchToSelectiveSearchQuality()
    # if argument is neither f nor q print help message
    
    rects = ss.process()
    # adapt the x, y, width, height format to x, y, x + width, y + height format
    new_rects = np.array([[rect[0], rect[1], rect[0] + rect[2], rect[1] + rect[3]] for rect in rects])
    # tqdm.write('Total Number of Region Proposals: {}'.format(len(rects)))

    return im, new_rects
    
def display_result(
        im, 
        rects, 
        numShowRects, # number of region proposals to show
        increment, # increment to increase/decrease total number of reason proposals to be shown
    ):
    '''run selective search segmentation on input image'''
 
    while True:
        # create a copy of original image
        imOut = im.copy()
 
        # itereate over all the region proposals
        for i, rect in enumerate(rects):
            # draw rectangle for region proposal till numShowRects
            if (i < numShowRects):
                x, y, w, h = rect
                cv2.rectangle(imOut, (x, y), (w, h), (0, 255, 0), 1, cv2.LINE_AA)
            else:
                break
 
        # show output
        print("{} proposals are shown.".format(numShowRects if numShowRects < len(rects) else len(rects)))
        cv2.imshow("Output", imOut)
 
        # record key press
        k = cv2.waitKey(0) & 0xFF
 
        # m is pressed
        if k == 109:
            # increase total number of rectangles to show by increment
            numShowRects = numShowRects + increment if numShowRects + increment < len(rects) + increment else numShowRects
        # l is pressed
        elif k == 108 and numShowRects > increment:
            # decrease total number of rectangles to show by increment
            numShowRects -= increment
        # q is pressed
        elif k == 113:
            break
    # close image show window
    cv2.destroyAllWindows()

def output_pickle(image_res, output_dir):
    indexes = []
    boxes = []
    scores = []
    for res in image_res:
        indexes.append(int(res[2].split('.')[0]))
        boxes.append(res[1])
        scores.append(np.ones(res[1].shape[1]))

    pkl_output = {'indexes': indexes, 'boxes': boxes, 'scores': scores}
    
    output_dir_parent_dir = os.path.dirname(output_dir)
    if not os.path.exists(output_dir_parent_dir):
        os.makedirs(output_dir_parent_dir)
    with open(output_dir, 'wb') as f:
        pickle.dump(pkl_output, f)

def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--show', action="store_true", help='Show results')
    parser.add_argument('--show-num', type=int, default=100, help='Number of region proposals to show')
    parser.add_argument('--show-increment', type=int, default=50, help='Increment to increase/decrease total number of reason proposals to be shown')
    parser.add_argument('-t', '--type', type=str, choices=['f', 'q'], default='q', help="'f' for speed-centered algorithm, 'q' for quality-centered algorithm")
    parser.add_argument('-p', '--path', type=str, help="Path of the image")
    parser.add_argument('-d', '--dir', type=str, help="Directory of the folder containing images")
    parser.add_argument('-ft', '--file-type', type=str, default='jpg',help="Extension type of images, used with -d/--dir; separate with ',' like jpg,png")
    parser.add_argument('-i', '--input-index', type=str, help="Path of the txt file containing index of pictures, used with -d/--dir")
    parser.add_argument('-o', '--output', type=str, help="Path of the pkl file output, used with -d/--dir")
    return parser.parse_args()

def main():
    args = arg_parse()
    if args.path is not None:
        if args.dir is not None:
            raise Exception("Only one of the image path or folder path is permitted.")
        image, rects = selective_search(args.path)

        if args.show:
            print("Press m to increase rectangle numbers by {}.".format(args.show_increment))
            print("Press l to decrease rectangle numbers by {}.".format(args.show_increment))
            print("Press q to stop displaying current image.")
            display_result(image, rects, numShowRects=args.show_num, increment=args.show_increment)
    elif args.dir is not None:
        extensions = args.file_type.replace(' ', '').split(',')
        walk_res = os.walk(args.dir)

        if args.input_index is not None:
            with open(args.input_index, "r") as f:
                ids = f.read()
            id_list = [id for id in ids.split("\n") if len(id) > 0]

        img_infos = []
        for root, dirs, files in walk_res:
            for f in files:
                if '.' in f and f.split('.')[1] in extensions:
                    if args.input_index is None:
                        img_infos.append([f, os.path.join(root, f)])
                    elif f.split('.')[0] in id_list:
                        img_infos.append([f, os.path.join(root, f)])

        image_res = []
        for info in tqdm(img_infos, ascii=True):
            image, rects = selective_search(info[1])
            image_res.append([image, rects, info[0]])

        if args.show:
            print("Press m to increase rectangle numbers by {}.".format(args.show_increment))
            print("Press l to decrease rectangle numbers by {}.".format(args.show_increment))
            print("Press q to stop displaying current image.")
            for res in image_res:
                display_result(res[0], res[1], numShowRects=args.show_num, increment=args.show_increment)

        if args.output is not None:
            output_pickle(image_res, args.output)

    else:
        raise Exception("No image path or folder path is given.")
     
if __name__ == '__main__':
    # speed-up using multithreads
    cv2.setUseOptimized(True)
    cv2.setNumThreads(8)
    
    main()
 
 