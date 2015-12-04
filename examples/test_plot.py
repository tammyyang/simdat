from simdat.core import plot
pl = plot.PLOT()
#a = [[1,3,5,6,7], [2,4,2,4,2], [8,6,4,2,0], [5,6,7,3,4]]
N = 5
a = [[2, 3, 3, 5, 2],
     [2, 3, 4, 1, 2],
     [2, 3, 4, 1, 2],
     [2, 3, 4, 1, 2],
     [3, 5, 2, 3, 3]]
d = [30, 50, 15, 5]
#a = [[[8,6,4,2,0], [5,6,7,3,4]], [[1,3,5,6,7], [2,4,2,4,2]]] 
#pl.plot_2D_dists(a, clear=False, fname=None)
#pl.plot_stacked_bar(a, clear=False, fname=None)
#pl.histogram(d, clear=False, fname=None)
pl.plot_pie(d, expl=2, clear=False)
#pl.plot_matrix(a)
#pl.plot_single_bar(d, clear=False)
#x = (1000, 1200)
#y = (1500, 2000)
#pl.patch_line(x, y, clear=False, fname=None)
pl.patch_ellipse(0.5, 0.5, w=10, h=5, angle=45, clear=False, fname=None)
#pl.patch_rectangle(0, 0, w=1000, h=500, clear=False, fname=None)
#pl.patch_arrow(10, 20, width=100, fill=True, clear=False)
pl.patch_textbox(0.5, 0.5, 'I hate to plot')
