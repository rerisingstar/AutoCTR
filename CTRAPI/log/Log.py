import logging
import time

def set_logger(model_name, data_name, dir_name=None):
    if dir_name == None:
        time_now = time.strftime("%Y-%m-%d_%H-%M-%S",time.localtime())
        file_name = '_'.join([time_now, model_name, data_name])
    else:
        time_now = time.strftime("%Y-%m-%d_%H-%M-%S",time.localtime())
        file_name = '_'.join([time_now, model_name, data_name])
        file_name = dir_name+'/'+file_name

    logger = logging.getLogger(__name__)
    logger.setLevel(level=logging.INFO)
    import os
    dir_dir_dir_name = os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(os.path.realpath(__file__))))))
    handler = logging.FileHandler(dir_dir_dir_name+'/LOG/' + file_name + '.txt')
    #handler = logging.FileHandler(dir_dir_dir_name+'/LOG/' + 'ook1.txt')
    handler.setLevel(logging.INFO)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.addHandler(console)

    file_position = dir_dir_dir_name+'/LOG/' + file_name + '.txt'
    return logger, file_position

if __name__ == '__main__':

    # t = time.strftime("%Y-%m-%d %H:%M:%S",time.localtime())
    # print(t)

    # logger = logging.getLogger(__name__)
    # logger.setLevel(level=logging.INFO)
    # handler = logging.FileHandler("log.txt")
    # handler.setLevel(logging.INFO)
    # formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    # handler.setFormatter(formatter)
    #
    # console = logging.StreamHandler()
    # console.setLevel(logging.INFO)
    #
    # logger.addHandler(handler)
    # logger.addHandler(console)
    #

    logger = set_logger('DIEN', 'Amazon')

    logger.info("Start print log")
    logger.debug("Do something")
    logger.warning("Something maybe fail.")
    logger.info("Finish")