import os
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.visualizer import save_images
from util import html


if __name__ == '__main__':
    opt = TestOptions().parse()  # get test options
    # hard-code some parameters for test
    opt.num_threads = 0   # test code only supports num_threads = 1
    opt.batch_size = 1    # test code only supports batch_size = 1
    opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.

    if opt.stage == "0":
        opt.name += "_offset_prediction"
        #opt.dis_2 = False
    elif opt.stage == "1":
        opt.name += "_font_generation"
        opt.dis_2 = True
    elif opt.stage == "0,1":
        opt.stage = "0"
        #opt.dis_2 = False
        name = opt.name
        opt.name += "_offset_prediction"
        #continue_train = opt.continue_train
        #opt.continue_train = True
        model_offset = create_model(opt)      # create a model given opt.model and other options
        model_offset.setup(opt)               # regular setup: load and print networks; create schedulers
        opt.stage = "0,1"
        opt.dis_2 = True
        opt.name = name
        opt.name += "_font_generation"
        #opt.continue_train = continue_train

    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers
    # create a website
    web_dir = os.path.join(opt.results_dir, opt.name, '{}_{}'.format(opt.phase, opt.epoch))  # define the website directory
    if opt.load_iter > 0:  # load_iter is 0 by default
        web_dir = '{:s}_iter{:d}'.format(web_dir, opt.load_iter)
    print('creating web directory', web_dir)
    webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.epoch))
    # test with eval mode. This only affects layers like batchnorm and dropout.
    if not opt.no_eval:
        model.eval()
    for i, data in enumerate(dataset):
        if i >= opt.num_test:  # only apply our model to opt.num_test images.
            break
        if opt.stage != "0,1":
            model.set_input(data)         # unpack data from dataset and apply preprocessing
        else:
            model.set_input(data, model_offset)         # unpack data from dataset and apply preprocessing
        #model.set_input(data)  # unpack data from data loader
        model.test()           # run inference
        visuals = model.get_current_visuals()  # get image results
        img_path = model.get_image_paths()     # get image paths
        if i % 5 == 0:  # save images to an HTML file
            print('processing (%04d)-th image... %s' % (i, img_path))
        save_images(webpage, visuals, img_path, aspect_ratio=opt.aspect_ratio, width=opt.display_winsize, fid_mode=opt.fid_mode)
    #webpage.save()  # save the HTML
