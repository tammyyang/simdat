import os
import sys
import time
import json
import numpy as np


class NumpyAwareJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray) and obj.ndim == 1:
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


class TOOLS(object):
    def __init__(self):
        """Init function of TOOLS, class of small tools"""

        self.tools_init()
        return

    def path_suffix(self, path, level=2):
        """Return the last parts of the path with a given level"""

        splits = path.split('/')
        suf = splits[-1]
        for i in range(2, level+1):
            suf = os.path.join(splits[0-i], suf)
        return suf

    def read_template(self, fname, temp_vars):
        """Read jinja template

        @param fname: Inpur file name
        @temp_vars: Variable dictionary to be used for the template

        @return: Rendered template

        """
        from jinja2 import FileSystemLoader, Environment
        templateLoader = FileSystemLoader(searchpath="/")
        templateEnv = Environment(loader=templateLoader)
        try:
            template = templateEnv.get_template(fname)
            return template.render(temp_vars)
        except:
            print("Exception:", sys.exc_info()[0])
            raise

    def tools_init(self):
        pass

    def gen_md5(self, data):
        """Generate md5 hash for a data structure"""

        import hashlib
        import pickle
        _data = pickle.dumps(data)
        return hashlib.md5(_data).hexdigest()

    def print_time(self, t0, s):
        """Print how much time has been spent

        @param t0: previous timestamp
        @param s: description of this step

        """
        print("%.5f seconds to %s" % ((time.time() - t0), s))
        return time.time()

    def conv_to_date(self, raw_data, key):
        """Convert y,m,d assigned in json profile to date object"""

        from datetime import date, timedelta
        return date(raw_data[key]["y"],
                    raw_data[key]["m"], raw_data[key]["d"])

    def get_combinations(self, input_list, n=2):
        """Get all combination of elements of input list

        @param input_list: input array

        Keyword arguments:
        n -- size of a combination (default: 2)

        """
        from itertools import combinations_with_replacement
        return combinations_with_replacement(input_list, n)

    def check_exist(self, path):
        """Check if the path exists

        @param path: path to check

        """
        if os.path.exists(path):
            return True
        else:
            print("%s does not exist" % path)
            return False

    def check_ext(self, file_name, extensions):
        """Check the file extension

        @param file_name: input file name
        @param extensions: string or list, extension(s) to check

        @return bool: True if it is matched

        """
        if file_name.endswith(extensions):
            return True
        return False

    def check_yes(self, answer):
        """Check if the answer is yes"""
        if answer.lower() in ['y', 'yes']:
            return True
        return False

    def dir_check(self, dirpath):
        self.check_dir(dirpath)

    def check_dir(self, dirpath):
        """Check if a directory exists.
           create it if doean't"""

        if not os.path.exists(dirpath):
            print("Creating %s" % dirpath)
            os.makedirs(dirpath)

    def check_parent(self, fpath):
        """Check if the parent directory exists.
           create it if doean't"""

        dirname = os.path.dirname(fpath)
        self.check_dir(dirname)

    def move_file(self, fpath, newhome, ask=True):
        """Move a file

        @param fpath: the original path of the file
        @param newhome: new home (directory) of the file

        Keyword arguments:
        ask -- true to ask before moving (default: True)

        """
        import shutil
        dirname = os.path.dirname(fpath)
        self.check_dir(newhome)
        newpath = fpath.replace(dirname, newhome)
        if ask:
            yn = self.check_yes(raw_input('Do you want to move %s to %s?'
                                % (fpath, newhome)))
        else:
            yn = True
        if yn:
            shutil.move(fpath, newpath)


class MLIO(TOOLS):
    def write_csv(self, data, fname='output.csv'):
        """Write data to csv

        @param data: object to be written

        Keyword arguments:
        fname  -- output filename (default './output.csv')

        """
        import csv
        f = open(fname, 'wb')
        wr = csv.writer(f, dialect='excel')
        wr.writerows(data)

    def read_csv(self, fname, ftype=None):
        """Read CSV file as list

        @param fname: file to read

        Keyword arguments:
        ftype  -- convert data to the type (default: None)

        @return list of csv content

        """
        import csv
        output = []
        with open(fname, 'rt') as csvfile:
            for row in csv.reader(csvfile, delimiter=','):
                if ftype is not None:
                    row = map(ftype, row)
                output.append(list(row))
        return output

    def read_json_to_df(self, fname, orient='columns', np=False):
        """Read json file as pandas DataFrame

        @param fname: input filename

        Keyword arguments:
        orient -- split/records/index/columns/values (default: 'columns')
        np     -- true to direct decoding to numpy arrays (default: False)
        @return pandas DataFranm

        """
        import pandas as pd
        self.check_exist(fname)
        return pd.read_json(fname, orient=orient, numpy=np)

    def read_jsons_to_df(self, flist, orient='columns', np=False):
        """Read json files as one pandas DataFrame

        @param fname: input file list

        Keyword arguments:
        orient -- split/records/index/columns/values (default: 'columns')
        np     -- true to direct decoding to numpy arrays (default: False)
        @return concated pandas DataFranm

        """
        import pandas as pd
        dfs = []
        for f in flist:
            dfs.append(self.read_json_to_df(f, orient=orient, np=np))
        return pd.concat(dfs)

    def write_df_json(self, df, fname='df.json'):
        """Wtite pandas.DataFrame to json output"""

        df.to_json(fname)
        print('DataFrame is written to %s' % fname)

    def read_csv_to_np(self, fname='data.csv'):
        """Read CSV file as numpy array

        Keyword arguments:
        fname  -- input filename (default './data.csv')

        @return numpy array

        """
        self.check_exist(fname)
        content = self.read_csv(fname=fname, ftype=float)
        return DATA().conv_to_np(content)

    def parse_json(self, fname):
        """Parse the input profile

        @param fname: input profile path

        @return data: a dictionary with user-defined data for training

        """
        import json
        with open(fname) as data_file:
            data = json.load(data_file)
        return data

    def write_json(self, data, fname='./output.json'):
        """Write data to json

        @param data: object to be written

        Keyword arguments:
        fname  -- output filename (default './output.json')

        """
        import json
        with open(fname, 'w') as fp:
            json.dump(data, fp, cls=NumpyAwareJSONEncoder)

    def conv_csv_svmft(self, csvfile, target=0,
                       ftype=float, classify=True):
        """Convert csv file to SVM format

        @param csvfile: file name of the csv to be read

        Keyword arguments:

        target   -- target column (default: 0)
        ftype    -- convert data to the type (default: None)
        classify -- true convert target to int type (default: True)

        """
        dt = DATA()
        indata = self.read_csv(csvfile, ftype=ftype)
        df = dt.conv_to_df(indata)

        _data = df.drop(df.columns[[target]], axis=1)
        data = dt.conv_to_np(_data)
        target = dt.conv_to_np(df[target])

        self.write_svmft(target, data, classify=classify)

    def write_svmft(self, target, data, classify=True,
                    fname='./data.svmft'):
        """Output data with the format libsvm/wusvm accepts

        @param target: array of the target (1D)
        @param data: array of the data (multi-dimensional)

        Keyword arguments:
        classify -- true convert target to int type (default: True)
        fname    -- output file name (default: ./data.svmft)

        """

        length = DATA().check_len(target, data)
        if classify:
            target = TOOLS().conv_to_np(target)
            target = target.astype(int)

        with open(fname, 'w') as outf:
            for i in range(0, length):
                output = []
                output.append(str(target[i]))
                for j in range(0, len(data[i])):
                    output.append(str(j+1) + ':' + str(data[i][j]))
                output.append('\n')
                libsvm_format = ' '.join(output)
                outf.write(libsvm_format)


class DATA(TOOLS):
    def cal_vector_length(self, array):
        """Calculate the length of an input array"""

        import math
        array = self.conv_to_np(array)
        mean = np.square(array).mean()
        return math.sqrt(mean)

    def cal_standard_error(self, array):
        """Calculate standard error"""

        import math
        array = self.conv_to_np(array)
        return np.std(array)/math.sqrt(len(array))

    def conv_to_df(self, array, ffields=None, target=None):
        """Convert array to pandas.DataFrame

        @param array: input array to be converted

        Keyword arguments:
        ffields -- json file of the fields (default: None)
        target  -- if ffields is specified, can also specified
                   the target column to be used (default: None)

        """
        import pandas as pd
        if ffields is not None:
            fields = MLIO().parse_json(ffields)
            if type(target) is int:
                print('Converting field from %s to target'
                      % fields[target])
                fields[target] = 'target'
            return pd.DataFrame(array, columns=fields)
        return pd.DataFrame(array)

    def df_header(self, df):
        """Get the header of the DataFrame as a list"""

        header = df.columns.values.tolist()
        print('DataFrame header:')
        print(header)
        return header

    def check_len(self, a, b):
        """Check if two arrays have the same length"""

        la = len(a)
        lb = len(b)
        if la == lb:
            return la
        print("ERROR: length of a (%i) and b (%i) are different" % (la, lb))
        sys.exit(1)

    def get_perc(self, data):
        """Convert the input data to percentage of the total sum

        @param data: input data array (1D)

        @return data in percentage-wise

        """

        data = self.conv_to_np(data)
        data = data.astype(float)
        return (data/np.sum(data))*100

    def is_np(self, array):
        """Check if the input array is in type of np.ndarray"""

        if type(array) in [np.ndarray, np.int64]:
            return True
        return False

    def conv_to_np(self, array):
        """Convert DataFrame or list to np.ndarray"""

        from pandas.core.frame import DataFrame, Series
        if type(array) in [DataFrame, Series]:
            return array.as_matrix()

        if type(array) is list:
            return np.array(array)

        if self.is_np(array):
            return array

        print("WARNING: the type of input array is not correct!")
        print(type(array))
        return array


class IMAGE(TOOLS):
    def find_images(self, dir_path=None, keyword=None):
        """Find images under a directory


        Keyword arguments:
        dir_path -- path of the directory to check (default: '.')
        keyword  -- keyword used to filter images (default: None)

        @return output: a list of images found

        """
        if dir_path is None:
            dir_path = os.getcwd()
        output = []
        for dirPath, dirNames, fileNames in os.walk(dir_path):
            dirmatch = False
            if keyword is not None and dirPath.find(keyword) > 0:
                dirmatch = True
            for f in fileNames:
                if keyword is not None and dirPath.find(keyword) < 0:
                    if not dirmatch:
                        continue
                if self.check_ext(f, ('.jpg', 'png')):
                    output.append(os.path.join(dirPath, f))
        return output

    def get_img_info(self, img_path):
        """Find image size and pixel array

        @param img_path: path of the input image

        @return image.size: tuple, size of the image
        @return pix: pixel of the image

        """
        from PIL import Image
        im = Image.open(img_path)
        pix = im.load()
        return im.size, pix

    def get_images(self, path):
        """Find images from the given path"""

        if os.path.isfile(path):
            if self.check_ext(path, ('.jpg', 'png')):
                return [path]
        elif os.path.isdir(path):
            return self.find_images(path)

    def get_jpeg_quality(self, img_path):
        """Get the jpeg quality using identify tool"""

        import subprocess
        try:
            q = subprocess.check_output("identify -verbose %s | grep Quality"
                                        % img_path, shell=True)
            q = q.replace(' ', '').split('\n')[0].split(':')[1]
            return int(q)
        except subprocess.CalledProcessError:
            return None
