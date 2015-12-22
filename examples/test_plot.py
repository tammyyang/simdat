from simdat.core import plot
pl = plot.PLOT()
a = [[1, 2, 3, 4, 5],
     [2, 2, 2, 2, 2],
     [1, 1, 1, 1, 1],
     [1, 2, 3, 4, 5],
     [5, 4, 3, 2, 1]]

b = [[[1, 2, 3, 4, 5], [2, 2, 2, 2, 2]],
     [[-1, -1, -1, -1, -1],[5, 4, 3, 2, 1]]]

d = [30, 50, 15, 5]
# Example to draw multiple bars

# xticks = [2007, 2008, 2009, 2010, 2011,
#           2012, 2013, 2014, 2015]
# legend = ['Average Return(%)', 'Accuuracy(%)']
# a = [[-22.74,  27.52, -43.52, -13.71,  -46.11,  42.58, 16.84, 25.53, 1.97],
#      [64.4,  63,  63.7,  66.6,  65.8,  67,  66,  61,  62]]
# title = 'TW 2498 Average Revenue vs Accuracy'
# pl.plot_multi_bars(a, xticks=xticks, legend=legend, title=title,
#                    color='brown', leg_loc='c')

# Example to draw 1D distributions
# pl.plot_1D_dists(a, leg_loc='rt')

# Example to draw 2D distributions
# pl.plot_2D_dists(a, clear=False)

# Example to draw scatter plot of classified data
# pl.plot_classes(b)

# Example to draw stacked bars
# pl.plot_stacked_bar(a, color='brown')

# Example to draw pie chart
# pl.plot_pie(d, expl=2)

# Example to draw histogram chart
# pl.histogram(d)

# Example to draw matrix
# pl.plot_matrix(a, show_text=False, color='viridis')

# Example to draw bars with errors
# err = [1, 3, 2, 5]
# pl.plot_single_bar(d, err=err, xrotation=0, width=0.5)

# Example to patch a shape to the existing figure
# x = (1000, 1200)
# y = (1500, 2000)
# pl.patch_line(x, y, clear=False, fname=None)
# pl.patch_ellipse(0.5, 0.5, w=10, h=5, angle=45)
# pl.patch_rectangle(0, 0, w=1000, h=500, clear=False, fname=None)
# pl.patch_arrow(10, 20, width=100, fill=True, clear=False)
# pl.patch_textbox(0.5, 0.5, 'I hate to plot')

