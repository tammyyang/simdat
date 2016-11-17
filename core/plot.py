import sys
import os
from os import path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from matplotlib import font_manager
from simdat.core import tools


class COLORS:
    red = ['#7E3517', '#954535', '#8C001A', '#C11B17',
           '#C04000', '#F62217', '#E55B3C',
           '#E78A61', '#FAAFBE', '#FFDFDD']

    grey = ['#0C090A', '#2C3539', '#413839', '#504A4B',
            '#666362', '#646D7E', '#6D7B8D',
            '#837E7C', '#D1D0CE', '#E5E4E2']

    brown = ['#493D26', '#7F462C', '#7F5217', '#B87333',
             '#C58917', '#C7A317', '#FFDB58',
             '#FFE87C', '#FFFFC2', '#F5F5DC']

    green = ['#254117', '#306754', '#437C17', '#728C00',
             '#4E9258', '#41A317', '#4CC417',
             '#99C68E', '#B2C248', '#C3FDB8']

    pink = ['#7F525D', '#C12267', '#E45E9D', '#FC6C85',
            '#F778A1', '#E38AAE', '#E799A3',
            '#FBBBB9', '#FFDFDD', '#FCDFFF']

    blue = ['#4863A0', '#737CA1', '#488AC7', '#98AFC7',
            '#38ACE7', '#659EC7', '#79BAE7',
            '#A0CFEC', '#C6DEFF', '#BDEDFF']

    colors = ['red', 'grey', 'green', 'brown',
              'pink', 'blue']


class PLOT(tools.DATA, COLORS):
    def tools_init(self):
        self.ax = plt.axes()
        self.loc_map = {'rt': 1, 'rb': 4, 'lb': 3,
                        'lt': 2, 'c': 9, 'cb': 8, 'ct': 9}

    def check_array_length(self, arrays):
        """Check if lengths of all arrays are equal

        @param arrays: input list of arrays

        @return length: the length of all arrays

        """
        length = len(arrays[0])
        for a in arrays[1:]:
            if len(a) != length:
                print('[PLOT] ERROR: array lengths are not equal')
                sys.exit(1)
        return length

    def find_axis_max_min(self, values, s_up=0.1, s_down=0.1):
        """Find max and min values used for setting axis"""

        import math
        values = self.conv_to_np(values)

        axis_max = np.amax(values)
        axis_min = np.amin(values)
        if axis_max < 1 and axis_max >= 0:
            axis_max = axis_max + s_up
        else:
            axis_max = axis_max * (1 + self.sign(axis_max) * s_up)
        if axis_min > -1 and axis_min <= 0:
            axis_min = axis_min - s_down
        else:
            axis_min = axis_min * (1 - self.sign(axis_min) * s_down)
        return axis_max, axis_min

    def scale(self, a):
        """Use no.linalg.norm to normalize the numpy array"""

        a = self.conv_to_np(a)
        return a / np.linalg.norm(a)

    def open_img(self, imgpath, clear=False):
        """Open an image on panel

        @param imgpath: path of the image

        Keyword arguments:
        clear     -- true to clear panel after output (default: False)

        @return image object and (xmax, ymax)

        """
        import matplotlib.image as mpimg
        img = mpimg.imread(imgpath)
        xmax = len(img[0])
        ymax = len(img)
        print('[PLOT] x max = %i, y max = %i' % (xmax, ymax))
        plt.imshow(img)
        if clear:
            plt.clf()
        return img, (xmax, ymax)

    def _add_titles(self, title, xlabel, ylabel):
        """Add title, xlabel and ylabel to the figure"""

        plt.title(title, color='#504A4B', weight='bold')
        self.ax.set_ylabel(ylabel, color='#504A4B')
        self.ax.set_xlabel(xlabel, color='#504A4B')

    def _define_legend_args(self, loc, mandarin, size):
        """Define the argument for legend

        @param loc: location of the legend
                    rt   - right top
                    rb   - right bottom
                    lt   - left top
                    lb   - left bottom
                    c/ct - central top
                    cb   - central bottom
        """

        args = {'loc': self.loc_map[loc]}
        font_file = path.join(os.environ['HOME'], '.fonts/noto/',
                              'NotoSansCJKtc-Light.otf')
        if self.check_exist(font_file) and mandarin:
            chf = font_manager.FontProperties(fname=font_file)
            args["prop"] = chf
        elif type(size) is int:
            args["prop"] = {"size": size}
        if loc == 'rt':
            args['bbox_to_anchor'] = (1.12, 1.0)
        elif loc in ['rb', 'lb', 'lt']:
            args['borderaxespad'] = 1
        elif loc in ['c', 'ct']:
            args['borderaxespad'] = -2.5
        return args

    def _add_legend(self, loc, mandarin=False, size=None):
        """Add Legend to the figure"""

        args = self._define_legend_args(loc, mandarin, size)
        self.ax.legend(**args)

    def _add_legend_2D(self, plots, legend, size=None,
                       loc='rt', mandarin=False):
        """Add Legend to the 2D figure

        @param plots: a list of plot objects
        @param legend: a list of legend

        """

        args = self._define_legend_args(loc, mandarin, size)
        plt.legend(plots, legend, **args)

    def patch_line(self, x, y, color='#7D0552', clear=True,
                   linewidth=None, linestype='solid',
                   fname='./patch_line.png'):
        """Patch a line to the existing panel

        @param x: x data, [x1, x2] where x1 and x2 are x positions
        @param y: y data, [y1, y2] where y1 and y2 are y positions

        Keyword arguments:
        color     -- line color (default: #7D0552)
        clear     -- true to clear panel after output (default: True)
        linewidth -- width of the edge line
        linestype -- style of the edge, solid (default), dashed,
                     dashdot, dotted
        fname     -- output filename (default: './patch_line.png')

        """

        args = {'color': color,
                'linestyle': linestype}
        if linewidth is not None:
            args['linewidth'] = linewidth
        currentAxis = plt.gca()
        currentAxis.add_line(plt.Line2D(x, y, **args))
        if fname is not None:
            plt.savefig(fname)
        if clear:
            plt.clf()

    def patch_arrow(self, x, y, dx=20, dy=100, width=10,
                    color='#566D7E', fill=False, clear=True,
                    linewidth=None, linestype='dashed',
                    fname='./patch_arrow.png'):
        """Patch a arrow to the existing panel

        @param x: the starting point of x axis, should be a single value
        @param y: the starting point of y axis, should be a single value

        Keyword arguments:
        dx        -- delta x of the arrow
        dy        -- delta y of the arrow
        width     -- width of the arrow
        color     -- arrow color (default: #566D7E)
        fill      -- true to fill the arrow (defaule: False)
        clear     -- true to clear panel after output (default: True)
        linewidth -- width of the edge line
        linestype -- style of the edge, solid, dashed (default),
                     dashdot, dotted
        fname     -- output filename (default: './patch_arrow.png')

        """
        from matplotlib.patches import Arrow
        args = {'edgecolor': color,
                'width': width,
                'linestyle': linestype}
        if fill:
            args['facecolor'] = color
            args['linestyle'] = 'solid'
        else:
            args['facecolor'] = 'none'
        if linewidth is not None:
            args['linewidth'] = linewidth
        currentAxis = plt.gca()
        currentAxis.add_patch(Arrow(x, y, dx, dy, **args))
        if fname is not None:
            plt.savefig(fname)
        if clear:
            plt.clf()

    def patch_textbox(self, x, y, text, style='round',
                      textcolor='#565051', edgecolor='#565051',
                      clear=True, fname='./patch_textbox.png'):
        """Patch a textbox to the existing panel

        @param x: the starting point of x axis, should be a single value
        @param y: the starting point of y axis, should be a single value
        @param text: text to show

        Keyword arguments:
        style     -- style of the bbox (default: round)
        textcolor -- color of the text (default: #565051)
        edgecolor -- color of the edge (default: #565051)
        clear     -- true to clear panel after output (default: True)
        fname     -- output filename (default: './patch_textbox.png')

        """
        args = {'edgecolor': edgecolor,
                'boxstyle': style,
                'facecolor': 'none'}
        self.ax.text(x, y, text, color=textcolor, bbox=args)
        if fname is not None:
            plt.savefig(fname)
        if clear:
            plt.clf()

    def patch_circle(self, x, y, radius=3,
                     color='#E77471', fill=False, clear=True,
                     linewidth=None, linestype='dashed',
                     fname='./patch_circle.png'):
        """Patch a circle to the existing panel"""

        self.patch_ellipse(x, y, w=radius, h=radius,
                           color=color, fill=fill, clear=clear,
                           linewidth=linewidth, linestype=linestype,
                           fname=fname)

    def patch_ellipse(self, x, y, w=5, h=3, angle=0,
                      color='#E77471', fill=False, clear=True,
                      linewidth=None, linestype='dashed',
                      fname='./patch_ellipse.png'):
        """Patch a ellipse to the existing panel

        @param x: the starting point of x axis, should be a single value
        @param y: the starting point of y axis, should be a single value

        Keyword arguments:
        w         -- width of the ellipse
        h         -- height of the ellipse
        angle     -- rotation angle of the ellipse (default: 0)
        color     -- ellipse color (default: #E77471)
        fill      -- true to fill the ellipse (defaule: False)
        clear     -- true to clear panel after output (default: True)
        linewidth -- width of the edge line
        linestype -- style of the edge, solid, dashed (default),
                     dashdot, dotted
        fname     -- output filename (default: './patch_ellipse.png')

        """
        from matplotlib.patches import Ellipse
        args = {'edgecolor': color,
                'angle': angle,
                'linestyle': linestype}
        if fill:
            args['facecolor'] = color
            args['linestyle'] = 'solid'
        else:
            args['facecolor'] = 'none'
        if linewidth is not None:
            args['linewidth'] = linewidth
        currentAxis = plt.gca()
        currentAxis.add_patch(Ellipse((x, y), w, h, **args))
        if fname is not None:
            plt.savefig(fname)
        if clear:
            plt.clf()

    def patch_rectangle(self, x, y, w=3, h=3, angle=0,
                        color='#6AA121', fill=False, clear=True,
                        linewidth=None, linestype='dashed',
                        fname='./patch_rectangle.png'):
        """Patch a rectangle to the existing panel

        @param x: the starting point of x axis, should be a single value
        @param y: the starting point of y axis, should be a single value

        Keyword arguments:
        w         -- width of the rectangle (default: 3)
        h         -- height of the rectangle (default: 3)
        angle     -- rotation angle of the rectangle (default: 0)
        color     -- rectangle color (default: #6AA121)
        fill      -- true to fill the rectangle (defaule: False)
        clear     -- true to clear panel after output (default: True)
        linewidth -- width of the edge line
        linestype -- style of the edge, solid, dashed (default),
                     dashdot, dotted
        title     -- chart title (default: '')
        fname     -- output filename (default: './points.png')

        """
        from matplotlib.patches import Rectangle
        args = {'edgecolor': color,
                'linestyle': linestype}
        if fill:
            args['facecolor'] = color
            args['linestyle'] = 'solid'
        else:
            args['facecolor'] = 'none'
        if linewidth is not None:
            args['linewidth'] = linewidth
        currentAxis = plt.gca()
        currentAxis.add_patch(Rectangle((x, y), w, h, **args))
        if fname is not None:
            plt.savefig(fname)
        if clear:
            plt.clf()

    def patch_rectangle_img(self, img_path, pos, new_home=None):
        """Open an image and patch a rectangle to it

        @param img_path: path of the image
        @param pos: position list, [left, top, right, bottom]

        Keyword parameters:
        new_home -- parent directory of the patched image
                    (default: ori_name.replace('.jpg', '_patch.jpg'))

        """
        import cv2
        if not self.check_exist(img_path):
            return
        img = cv2.imread(img_path)
        if new_home is None:
            new_path = img_path.replace('.jpg', '_patch.jpg')
            new_path = new_path.replace('.png', '_patch.png')
        else:
            base = path.basename(img_path)
            dirname = path.basename(path.dirname(img_path))
            new_path = path.join(new_home, dirname)
            self.dir_check(new_path)
            new_path = path.join(new_path, base)
        cv2.rectangle(img, (pos[0], pos[1]),
                      (pos[2], pos[3]), (0, 255, 0), 2)
        cv2.imwrite(new_path, img)

    def plot(self, data, clear=True, fname='./plot.png',
             title='', connected=True, ymax=None, log=False,
             ymin=None, xlabel='', ylabel='', xticks=None,
             xrotation=45, color=None, xmax=None, rebin=None):
        """Draw the very basic 1D plot

        @param data: an 1D array [y1, y2, y3...yn]

        Keyword arguments:
        clear     -- true to clear panel after output (default: True)
        xlabel    -- label of the X axis (default: '')
        ylabel    -- label of the y axis (default: '')
        title     -- chart title (default: 'Distributions')
        connected -- true to draw line between dots (default: True)
        xmax      -- maximum of x axis (default: max(data)+0.1)
        log       -- true to draw log scale (default: False)
        ymax      -- maximum of y axis (default: max(data)+0.1)
        ymin      -- minimum of y axis (default: max(data)-0.1)
        fname     -- output filename (default: './dist_1d.png')

        """

        data = self.conv_to_np(data)

        fmt = '-o' if connected else 'o'
        color = self.blue[2] if color is None else color
        plt.plot(data, fmt, color=color)
        _ymax, _ymin = self.find_axis_max_min(data)
        ymax = _ymax if ymax is None else max(ymax, _ymax)
        ymin = _ymin if ymin is None else min(ymin, _ymin)
        xmax = 1.1*(len(data)-1) if xmax is None else xmax

        plt.axis([-0.1, xmax, ymin, ymax])
        xtick_marks = np.arange(len(data))
        if log:
            self.ax.set_yscale('log')
        if xticks is None:
            xticks = xtick_marks
        if len(xticks) > 20 and rebin is None:
            rebin = len(xticks)/20
        if rebin is not None:
            xtick_marks, xticks = self.red_ticks(xtick_marks, xticks, rebin)
        plt.xticks(xtick_marks, xticks, rotation=xrotation)
        self._add_titles(title, xlabel, ylabel)
        if fname is not None:
            plt.savefig(fname)
        if clear:
            plt.clf()

    def plot_classes(self, data, fname='./classes.png',
                     xlabel='', ylabel='', legend=None, marker_size=40,
                     title='Classes', clear=True, leg_loc='rt'):
        """Plot scatter figures for multiple classes

        @param data: data [a1, a2, .., an] where a1, a2, an are 2D arrays
                     a1 = [[x1, y1], [x1, y2], ..., [xn, yn]]
                     xn, yn are 1D arrays of the x, y values in n class

        Keyword arguments:
        legend      -- a list of the legend, must match len(data)
                       (default: index of the list to be drawn)
        xlabel      -- label of the X axis (default: '')
        ylabel      -- label of the y axis (default: '')
        marker_size -- size of the markers
        clear       -- true to clear panel after output (default: True)
        title       -- chart title (default: '')
        fname       -- output filename (default: './points.png')

        """
        for i in range(0, len(data)):
            color_idx = i % len(self.colors)
            dks = i % 10
            color = getattr(self, self.colors[color_idx])[dks]
            args = {'c': color, 's': marker_size,
                    'edgecolors': 'face', 'alpha': 0.8}
            args['label'] = legend[i] if legend is not None else str(i)
            plt.scatter(data[i][0], data[i][1], **args)
        self._add_legend(leg_loc)
        self._add_titles(title, xlabel, ylabel)
        if fname is not None:
            plt.savefig(fname)
        if clear:
            plt.clf()

    def plot_pie(self, data, bfrac=False, shadow=False, clear=True,
                 title='Pie Chart', color='pink', radius=1.1,
                 pie_labels=None, expl=None, fname='./pie.png',
                 show_legend=False, show_frac=True, show_label=True):
        """Draw a pie chart

        @param data: a list of input data [x1, x2, x3,...,xn]

        Keyword arguments:
        bfrac      -- true if the input data already represents fractions
                      (default: False)
        shadow     -- add shadow to the chart (default: False)
        title      -- chart title (default: 'Pie Chart')
        color      -- color group to be used (default: 'pink')
        radius     -- radius of the pie (default: 1.1)
        pie_labels -- labels of each components
                      (default: index of the elements)
        expl       -- index of the item to explode (default: None)
        fname      -- output filename (default: './pie.png')
        show_frac  -- True to show fraction (default: True)
        show_legend  -- True to show fraction (default: False)
        show_label -- True to show labels (default: True)
        clear      -- true to clear panel after output (default: True)

        """

        data = self.conv_to_np(data)

        plt.figure(1, figsize=(6, 6))
        ax = plt.axes([0.1, 0.1, 0.8, 0.8])
        fracs = data if bfrac else self.get_perc(data)
        if pie_labels is None:
            pie_labels = list(map(str, range(1, len(data)+1)))
        color_class = getattr(self, color)

        args = {'shadow': shadow,
                'radius': radius,
                'textprops': {'color': color_class[0]},
                'startangle': 140,
                'colors': color_class[-len(data):]}

        if expl is not None:
            explode = [0]*len(data)
            explode[expl] = 0.05
            args['explode'] = explode

        if show_label:
            args['labels'] = pie_labels

        if show_frac:
            args['autopct'] = '%1.1f%%'

        if show_legend:
            patches, texts = plt.pie(fracs, **args)
            plt.legend(patches, pie_labels, loc="best")
        else:
            plt.pie(fracs, **args)

        plt.title(title, color='#504A4B', weight='bold')
        if fname is not None:
            plt.savefig(fname)
        if clear:
            plt.clf()

    def plot_stacked_bar(self, data, xticks=None, xlabel='', legend=None,
                         ylabel='', xrotation=45, width=0.6, clear=True,
                         color='blue', title='Stacked Bar Chart',
                         log=False, fname='stack_bar.png', leg_loc='rt'):
        """Draw a bar chart with errors

        @param data: a list of input data
                     [[x1, x2..xn], [y1, y2..yn], [z1, z2..zn],..]
                     where x1, y1, z1 are quantities at the first position

        Keyword arguments:
        xticks    -- ticks of the x axis (default: array index of the elements)
        xlabel    -- label of the X axis (default: '')
        legend    -- a list of the legend, must match len(data)
                     (default: index of the list to be drawn)
        ylabel    -- label of the y axis (default: '')
        xrotation -- rotation angle of xticks (default: 45)
        width     -- relative width of the bar (default: 0.6)
        color     -- color group to be used (default: 'blue')
        title     -- chart title (default: 'Stacked Bar Chart')
        log       -- true to draw log scale (default: False)
        fname     -- output filename (default: './stack_bar.png')
        clear     -- true to clear panel after output (default: True)

        """
        _len = self.check_array_length(data)
        data = self.conv_to_np(data)

        ind = np.arange(_len)
        stack_colors = getattr(self, color)
        if xticks is None:
            xticks = list(map(str, range(1, _len+1)))

        ymax = 0
        ymin = 0
        sum_array = np.zeros(_len)
        for i in range(0, len(data)):
            a = np.array(data[i]) if type(data[i]) is list else data[i]
            label = legend[i] if legend is not None else str(i)
            p = plt.bar(ind, a, width, bottom=sum_array,
                        log=log, label=label, alpha=0.85,
                        color=stack_colors[i % 10])
            sum_array = np.add(sum_array, a)
            _ymax, _ymin = self.find_axis_max_min(a)
            ymax += _ymax

        plt.axis([0, _len, ymin, ymax])
        self._add_legend(leg_loc)
        self._add_titles(title, xlabel, ylabel)
        plt.xticks(ind + width/2., xticks)
        if fname is not None:
            plt.savefig(fname)
        if clear:
            plt.clf()

    def plot_multi_bars(self, data, xticks=None, xlabel='', legend=None,
                        ylabel='', err=None, xrotation=45, clear=True,
                        color='green', title='Bar Chart', ecolor='brown',
                        log=False, fname='multi_bars.png', leg_loc='rt'):
        """Draw a bar chart with errors

        @param data: a 2D list of input data
                     [a1, a2, a3...]
                     a1 is a 1D array, a1 = [x1, x2, ...]
                     where the length of a1, a2, a3 should be the same

        Keyword arguments:
        xticks    -- ticks of the x axis (default: array index of the elements)
        xlabel    -- label of the X axis (default: '')
        ylabel    -- label of the y axis (default: '')
        legend    -- legend of the items (default: data index)
        err       -- upper error array (default: None)
        xrotation -- rotation angle of xticks (default: 45)
        clear     -- true to clear panel after output (default: True)
        color     -- color group (default: 'grey')
        title     -- chart title (default: 'Bar Chart')
        log       -- true to draw log scale (default: False)
        fname     -- output filename (default: './multi_bars.png')
        ecolor    -- color group of the error bars (default: 'brown')
        leg_loc   -- location of the legend, rt(default)/rb/lt/lb/c

        """

        data = self.conv_to_np(data)
        if xticks is None:
            xticks = list(map(str, range(1, len(data)+1)))

        ind = np.arange(len(xticks))
        N = len(data)
        width = 1.0/N

        for i in range(0, N):
            label = legend[i] if legend is not None else str(i)
            args = {'color': getattr(self, color)[i],
                    'label': label, 'alpha': 0.8}
            if err is not None:
                args['ecolor'] = getattr(self, ecolor)[i]
                args['yerr'] = err[i]
            _ind = ind + width * i * 0.9
            _rec = self.ax.bar(_ind, data[i], width*0.9, **args)

        if legend is None:
            legend = range(0, N)
        self._add_legend(leg_loc)
        self._add_titles(title, xlabel, ylabel)
        self.ax.set_xticks(ind+width/2)
        self.ax.set_xticklabels(xticks, rotation=xrotation)
        if log:
            self.ax.set_yscale('log')
        if fname is not None:
            plt.savefig(fname)
        if clear:
            plt.clf()

    def plot_single_bar(self, data, xticks=None, xlabel='',
                        ylabel='', err=None, xrotation=45, clear=True,
                        width=0.6, color='#FFCCCC', title='Bar Chart',
                        log=False, fname='bar.png', ecolor='#009966'):
        """Draw a bar chart with errors

        @param data: a list of input data
                     [x1, x2, x3...]
                     where x1, x2, x3 are quantities at the every position

        Keyword arguments:
        xticks    -- ticks of the x axis (default: array index of the elements)
        xlabel    -- label of the X axis (default: '')
        ylabel    -- label of the y axis (default: '')
        err       -- upper error array (default: None)
        xrotation -- rotation angle of xticks (default: 45)
        clear     -- true to clear panel after output (default: True)
        width     -- relative width of the bar (default: 0.6)
        color     -- color of the points (default: '#FFCCCC')
        title     -- chart title (default: 'Bar Chart')
        log       -- true to draw log scale (default: False)
        fname     -- output filename (default: './points.png')
        ecolor    -- color of the errors (default: '#00CCFF')

        """

        data = self.conv_to_np(data)
        if xticks is None:
            xticks = list(map(str, range(1, len(data)+1)))

        args = {'color': color, 'ecolor': ecolor}
        if err is not None:
            args['yerr'] = err

        ind = np.arange(len(xticks))
        rects1 = self.ax.bar(ind, data, width, **args)

        plt.title(title, color='#504A4B', weight='bold')
        self.ax.set_ylabel(ylabel, color='#504A4B')
        self.ax.set_xlabel(xlabel, color='#504A4B')
        self.ax.set_xticks(ind+width/2)
        self.ax.set_xticklabels(xticks, rotation=xrotation)
        if log:
            self.ax.set_yscale('log')
        if fname is not None:
            plt.savefig(fname)
        if clear:
            plt.clf()

    def plot_2D_dists(self, data, scale=False, legend=None, clear=True,
                      title='Distrubitions', connected=True, amin=None,
                      amax=None, xlabel='', ylabel='', yticks=None,
                      xmin=None, xmax=None, yrotation=0,
                      fname='./dist_2d.png', leg_loc='rt'):
        """Draw the dist of multiple 2D arrays.

        @param data: list of 2D arrays
                     [a1, a2, .., an] where a1, a2, an are 2D arrays
                     a1 = [[x1, x2...xn], [y1, y2...yn]]

        Keyword arguments:
        scale     -- true to scale the distributions (default: False)
        legend    -- a list of the legend, must match len(data)
                     (default: index of the list to be drawn)
        xlabel    -- label of the X axis (default: '')
        ylabel    -- label of the y axis (default: '')
        yticks    -- ticks of the y axis (default: array index)
        yrotation -- totation angle of y ticks (default: 0)
        clear     -- true to clear panel after output (default: True)
        title     -- chart title (default: 'Distributions')
        amax      -- maximum of y axis (default: max(data)+0.1)
        amin      -- minimum of y axis (default: max(data)-0.1)
        xmax      -- maximum of x axis (default: max(data)+0.1)
        xmin      -- minimum of x axis (default: max(data)-0.1)
        connected -- true to draw line between dots (default: True)
        fname     -- output filename (default: './dist_2d.png')

        """
        _len = self.check_array_length(data)
        data = self.conv_to_np(data)

        ymax = amax
        ymin = amin
        xmax = xmax
        xmin = xmin
        fmt = '-o' if connected else 'o'
        pls = []
        for i in range(0, len(data)):
            label = legend[i] if legend is not None else str(i)
            a = data[i]
            if type(a) is list:
                a = np.array(a)
            if scale:
                a = self.scale(a)

            p, = plt.plot(a[0], a[1], fmt, label=label)
            pls.append(p)

            _xmax, _xmin = self.find_axis_max_min(a[0])
            _ymax, _ymin = self.find_axis_max_min(a[1])
            xmax = _xmax if xmax is None else max(xmax, _xmax)
            xmin = _xmin if xmin is None else min(xmin, _xmin)
            if amax is None:
                ymax = _ymax if ymax is None else max(ymax, _ymax)
            if amin is None:
                ymin = _ymin if ymin is None else min(ymin, _ymin)

        plt.axis([xmin, xmax, ymin, ymax])

        if legend is not None:
            self._add_legend_2D(pls, legend, loc=leg_loc)

        if yticks is not None:
            ytick_marks = np.arange(len(yticks))
            plt.yticks(ytick_marks, yticks, rotation=yrotation)

        self._add_titles(title, xlabel, ylabel)
        if fname is not None:
            plt.savefig(fname)
        if clear:
            plt.clf()

    def plot_1D_dists(self, data, scale=False, legend=None, clear=True,
                      title='Distrubitions', connected=True, ymax=None,
                      ymin=None, xlabel='', ylabel='', xticks=None,
                      xrotation=45, leg_loc='rt', xmax=None,
                      leg_size=None, log=False, mandarin=False,
                      fname='./dist_1d.png', rebin=None):
        """Draw the dist of multiple 1D arrays.

        @param data: list of 1D arrays
                     [a1, a2, .., an] where a1, a2, an are 1D arrays
                     a1 = [x1, x2...xn]

        Keyword arguments:
        scale     -- true to scale the distributions (default: False)
        legend    -- a list of the legend, must match len(data)
                     (default: index of the list to be drawn)
        clear     -- true to clear panel after output (default: True)
        xlabel    -- label of the X axis (default: '')
        ylabel    -- label of the y axis (default: '')
        title     -- chart title (default: 'Distributions')
        connected -- true to draw line between dots (default: True)
        xmax      -- maximum of x axis (default: max(data)+0.1)
        ymax      -- maximum of y axis (default: max(data)+0.1)
        ymin      -- minimum of y axis (default: max(data)-0.1)
        fname     -- output filename (default: './dist_1d.png')
        rebin     -- N bins to be grouped together
        log       -- true to draw log scale (default: False)

        """

        data = self.conv_to_np(data)

        fmt = '-o' if connected else 'o'
        if rebin is not None:
            data = self.rebin2D(data, (data.shape[0], data.shape[1]/rebin))

        _ymax = None
        _ymin = None
        _xmax = None

        for i in range(0, len(data)):
            label = legend[i] if legend is not None else str(i)
            a = data[i]
            if type(a) is list:
                a = np.array(a)
            if scale:
                a = self.scale(a)

            color_idx = i % len(self.colors)
            dks = i % 10
            color = getattr(self, self.colors[color_idx])[dks]
            plt.plot(a, fmt, label=label, color=color)

            __ymax, __ymin = self.find_axis_max_min(a)
            __xmax = 1.1*(a.shape[0] - 1)
            _ymax = __ymax if _ymax is None else max(_ymax, __ymax)
            _ymin = __ymin if _ymin is None else min(_ymin, __ymin)
            _xmax = __xmax if _xmax is None else max(_xmax, __xmax)

        ymax = _ymax if ymax is None else ymax
        ymin = _ymin if ymin is None else ymin
        xmax = _xmax if xmax is None else xmax

        if log:
            self.ax.set_yscale('log')

        plt.axis([-0.1, xmax, ymin, ymax])
        xtick_marks = np.arange(len(data[0]))
        if xticks is None:
            xticks = xtick_marks
        if len(xticks) > 20 and rebin is None:
            rebin = len(xticks)/20
        if rebin is not None:
            xtick_marks, xticks = self.red_ticks(xtick_marks, xticks, rebin)
        plt.xticks(xtick_marks, xticks, rotation=xrotation)

        self._add_legend(leg_loc, mandarin=mandarin, size=leg_size)
        self._add_titles(title, xlabel, ylabel)
        if fname is not None:
            plt.savefig(fname)
        if clear:
            plt.clf()

    def diff_axis_1D(self, data, legend=None, c1=None, c2=None,
                     xrotation=45, connected=True, xticks=None,
                     clear=True, xmax=None, leg_loc='rt',
                     xlabel='', ylabel='', title='Distrubitions',
                     fname='./diff_axis_1D.png'):
        """Draw two dists with different axis

        @param data: [a1, a2] where a1 and a2 are 1D arrays
                     a1 = [x1, x2, .., xn]
                     a2 = [y1, y2..., yn]

        Keyword arguments:
        legend    -- a list of the legend, must match len(data)
                     (default: index of the list to be drawn)
        clear     -- true to clear panel after output (default: True)
        xrotation -- rotation angle of the xticks( default: 45)
        connected -- true to draw line between dots (default: True)
        xticks    -- xticks (default: data index)
        xlabel    -- label of the X axis (default: '')
        ylabel    -- label of the y axis (default: '')
        title     -- chart title (default: 'Distributions')
        fname     -- output filename (default: './dist_two.png')

        """

        c1 = self.red[4] if c1 is None else c1
        c2 = self.green[4] if c2 is None else c2
        legend = ['dist 1', 'dist 2'] if legend is None else legend
        fm = 'o-' if connected else 'o'
        xmax = len(data[0]) if xmax is None else xmax

        p1, = self.ax.plot(data[0], fm, color=c1, alpha=0.8, label=legend[0])
        for tl in self.ax.get_yticklabels():
            tl.set_color(c1)
        ymax1, ymin1 = self.find_axis_max_min(data[0])
        self.ax.set_ylim([ymin1, ymax1])
        self.ax.set_xlim([-0.5, xmax])

        ax2 = self.ax.twinx()
        p2, = ax2.plot(data[1], fm, color=c2, alpha=0.8, label=legend[1])
        for tl in ax2.get_yticklabels():
            tl.set_color(c2)
        ymax2, ymin2 = self.find_axis_max_min(data[1])
        ax2.set_ylim([ymin2, ymax2])
        ax2.set_xlim([-0.5, xmax])

        xtick_marks = np.arange(len(data[0]))
        if xticks is None:
            xticks = xtick_marks
        plt.xticks(xtick_marks, xticks, rotation=xrotation)
        self._add_titles(title, xlabel, ylabel)

        plt.legend([p1, p2], legend, loc=self.loc_map[leg_loc])
        if fname is not None:
            plt.savefig(fname)
        if clear:
            plt.clf()

    def histogram(self, data, xlabel='', ylabel='', clear=True,
                  title='Histogram', nbins=None, bfit=False, norm=False,
                  xlim=None, ylim=None, fname='./hist.png', grid=True,
                  align='mid', log=False, facecolor='#339966'):
        """Draw histogram of the numpy array

        @param data: input array (1D)
                     [x1, x2...xn] where xi are raw values

        Keyword arguments:
        xlabel -- label of the X axis (default: '')
        ylabel -- label of the y axis (default: '')
        clear  -- true to clear panel after output (default: True)
        xlim   -- limits of x axis (default: max, min of data)
        ylim   -- limits of y axis (default: max, min of data)
        norm   -- True to normalize to the first bin (default: False)
        title  -- chart title (default: 'Histogram')
        nbins  -- number of bins (default: length of the set of input data)
        bfit   -- also draw fit function (default: False)
        fname  -- output filename (default: './hist.png')
        grid   -- draw grid (default: True)
        align  -- histogram alignment, mid, left, right (default: mid)
        log    -- true to draw log scale (default: False)
        facecolor -- color of the histogram (Default: #339966)

        """
        data = self.conv_to_np(data)

        if nbins is None:
            nbins = len(set(data))
        y, x, patches = plt.hist(data, nbins, normed=norm, log=log,
                                 facecolor=facecolor, alpha=0.7,
                                 align=align, rwidth=1.0)
        if bfit:
            mu = np.mean(data)
            sigma = np.std(data)
            fit = mlab.normpdf(x, mu, sigma)
            plt.plot(x, fit, 'r--')

        plt.title(title, color='#504A4B', weight='bold')
        plt.ylabel(ylabel, color='#504A4B')
        plt.xlabel(xlabel, color='#504A4B')
        plt.grid(grid)
        if xlim is None:
            xmax, xmin = self.find_axis_max_min(x)
        else:
            xmin, xmax = xlim
        if ylim is None:
            ymax, ymin = self.find_axis_max_min(y)
        else:
            ymin, ymax = ylim
        plt.axis([xmin, xmax, 0, ymax])
        if fname is not None:
            plt.savefig(fname)
        if clear:
            plt.clf()

    def plot_points(self, x, y, err=None, err_low=None, clear=True,
                    connected=False, xlabel='', ylabel='', xticks=None,
                    fname='./points.png', title='', ymax=None, ymin=None,
                    xmax=None, xmin=None, ecolor='#3399FF', color='#CC6600'):
        """Plot points with (asymmetry) errors

        @param x: x array [x1, x2,...xn]
        @param y: y array [y1, y2,...yn]

        Keyword arguments:
        err       -- upper error array (default: None)
                     [e1, e2,...en]
        err_low   -- lower error array (default: None or err if err is set)
        connected -- true to draw line between dots (default: False)
        xticks     -- ticks of the x axis (default: array index)
        xlabel    -- label of the X axis (default: '')
        ylabel    -- label of the y axis (default: '')
        clear     -- true to clear panel after output (default: True)
        title     -- chart title (default: '')
        fname     -- output filename (default: './points.png')
        xmax      -- maximum of x axis (default: max(data)+0.1)
        xmin      -- minimum of x axis (default: max(data)-0.1)
        ymax      -- maximum of y axis (default: max(data)+0.1)
        ymin      -- minimum of y axis (default: max(data)-0.1)
        ecolor    -- color of the errors (default: '#3399FF')
        color     -- color of the points (default: '#CC6600')

        """
        x = self.conv_to_np(x)
        y = self.conv_to_np(y)

        fmt = '-o' if connected else 'o'
        args = {'fmt': fmt, 'ecolor': ecolor, 'color': color}
        if err is not None:
            if err_low is not None:
                args['yerr'] = [err_low, err]
            else:
                args['yerr'] = [err, err]
        self.ax.errorbar(x, y, **args)
        self.ax.set_title(title)
        _xmax, _xmin = self.find_axis_max_min(x)
        xmax = _xmax if xmax is None else xmax
        xmin = _xmin if xmin is None else xmin
        plt.xlim(xmin, xmax)
        _ymax, _ymin = self.find_axis_max_min(y)
        ymax = _ymax if ymax is None else ymax
        ymin = _ymin if ymin is None else ymin
        plt.ylim(ymin, ymax)
        self.ax.set_xlabel(xlabel)
        self.ax.set_ylabel(ylabel)
        if xticks is not None:
            xtick_marks = np.arange(len(x))
            plt.xticks(xtick_marks, xticks, rotation=45)
        if fname is not None:
            plt.savefig(fname)
        if clear:
            plt.clf()

    def plot_bubble_chart(self, x, y, z=None, scaler=1,
                          ascale_min=0.5, ascale_max=0.5,
                          xticks=None, xlabel='Bubble Chart',
                          clear=True, title='Bubble Chart',
                          ylabel='', fname='./bubble.png'):
        """Plot bubble chart

        @param x: x array, x positions of the bubbles
        @param y: y array, y positions of the bubbles

        Keyword arguments:
        z      -- z array to determine the size of bubbles (default [3]*N)
        scaler -- used for scaling the area (default: 1)
        ascale_min -- used for scaling x axis (default: 0.5)
        ascale_max -- used for scaling x axis (default: 0.5)
        title  -- chart title (default: 'Bubble Chart')
        xlabel -- label of the X axis (default 'Bubble Chart')
        ylabel -- label of the y axis (default '')
        fname  -- output filename (default './bubble.png')
        clear  -- true to clear panel after output (default: True)

        """
        N = len(x)
        x = self.conv_to_np(x)
        y = self.conv_to_np(y)
        colors = np.random.rand(N)
        if z is None:
            z = np.array([3]*N)
        elif type(z) is list:
            z = np.array(z)

        area = np.pi * (scaler * z)**2
        plt.scatter(x, y, s=area, c=colors, alpha=0.5)
        plt.ylabel(ylabel)
        plt.xlabel(xlabel)
        xlim = plt.xlim()
        plt.xlim(xlim[0]*ascale_min, xlim[1]*ascale_max)
        if xticks is not None:
            xtick_marks = np.arange(N)
            plt.xticks(xtick_marks, xticks, rotation=45)
        if fname is not None:
            plt.savefig(fname)
        if clear:
            plt.clf()

    def plot_confusion_matrix(self, cm, title='Confusion matrix',
                              xticks=None, yticks=None, fname='./cm.png',
                              xlabel='Predicted label',
                              ylabel='True label',
                              xrotation=45,
                              show_axis=True,
                              show_text=True, clear=True,
                              color='Blues', norm=True):

        self.plot_matrix(cm, title=title, xticks=xticks, yticks=yticks,
                         fname=fname, xlabel=xlabel, ylabel=ylabel,
                         xrotation=xrotation, show_text=show_text,
                         color=color, norm=norm, clear=clear,
                         show_axis=show_axis)

    def plot_matrix(self, cm, title='',
                    xticks=None, yticks=None, fname='./cm.png',
                    xlabel='Predicted label', ylabel='True label',
                    xrotation=45, clear=True,
                    color='YlOrRd', rebin=None, autorebin=False,
                    show_text=True, show_axis=True, norm=True):
        """Plot (confusion) matrix

        @param cm: input matrix (2D)
                   [a1, a2...an]
                   a = [x1, x2...xn]

        Keyword arguments:
        title      -- chart title (default: '')
        xticks     -- ticks of the x axis (default: array index)
        yticks     -- ticks of the y axis (default: array index)
        fname      -- output filename (default: './cm.png')
        xlabel     -- label of the X axis (default: 'Predicted label')
        ylabel     -- label of the y axis (default: 'True label')
        xrotation  -- rotation angle of xticks (default: 45)
        clear      -- true to clear panel after output (default: True)
        show_text  -- true to show values on grids (default: True)
        show_axis  -- true to show axis (default: True)
        color      -- color map, see http://goo.gl/51s91K (default: YlOrRd)
        autorebin  -- rebin automatically (default: False)
        norm       -- true to normlize numbers (default: True)

        """
        cm = self.conv_to_np(cm)
        if norm:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        cmap = getattr(plt.cm, color)
        plt.imshow(cm, interpolation='nearest', cmap=cmap, alpha=0.7)
        if show_text:
            diff = 1
            ind_array_x = np.arange(0, len(cm[0]), diff)
            ind_array_y = np.arange(0, len(cm), diff)
            x, y = np.meshgrid(ind_array_x, ind_array_y)
            for x_val, y_val in zip(x.flatten(), y.flatten()):
                c = round(cm[y_val][x_val], 2)
                self.ax.text(x_val, y_val, c, va='center', ha='center')

        plt.title(title, color='#504A4B', weight='bold')
        plt.colorbar()
        xtick_marks = np.arange(len(cm[0]))
        ytick_marks = np.arange(len(cm))
        if xticks is None:
            xticks = xtick_marks
        if yticks is None:
            yticks = ytick_marks
        if (len(xticks) > 20 or len(yticks) > 20) and autorebin:
            rebin = max(len(xticks), len(yticks))/20
        if rebin is not None:
            xtick_marks, xticks = self.red_ticks(xtick_marks, xticks, rebin)
            ytick_marks, yticks = self.red_ticks(ytick_marks, yticks, rebin)
        plt.xticks(xtick_marks, xticks, rotation=xrotation)
        plt.yticks(ytick_marks, yticks)

        plt.tight_layout()
        plt.ylabel(ylabel, color='#504A4B')
        plt.xlabel(xlabel, color='#504A4B')
        if not show_axis:
            plt.axis('off')
        if fname is not None:
            plt.savefig(fname)
        if clear:
            plt.clf()

    def red_ticks(self, marks, ticks, interval):
        orilen = len(ticks)
        nbins = orilen/interval
        if nbins < 1:
            return 0
        newmarks = [marks[0]]
        newticks = [ticks[0]]
        for i in range(0, len(marks)):
            if (i+1) % interval == 0:
                newmarks.append(marks[i])
                newticks.append(ticks[i])
        if orilen % nbins != 1:
            newmarks.append(marks[-1])
            newticks.append(ticks[-1])
        return newmarks, newticks
