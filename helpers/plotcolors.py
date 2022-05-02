import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors


myblue = (0.2980392156862745, 0.4470588235294118, 0.6901960784313725)
mygreen = (0.3333333333333333, 0.6588235294117647, 0.40784313725490196)
myred = (0.7686274509803922, 0.3058823529411765, 0.3215686274509804)
mypurple = (0.5058823529411764, 0.4470588235294118, 0.6980392156862745)
myyellow = (0.8, 0.7254901960784313, 0.4549019607843137)
myblue2 = (0.39215686274509803, 0.7098039215686275, 0.803921568627451)
myred2 = (1.0, 0.6235294117647059, 0.6039215686274509)

myblue_hex = u'#4c72b0'
mygreen_hex = u'#55a868'
myred_hex = u'#c44e52'
mypurple_hex = u'#8172b2'
myyellow_hex = u'#ccb974'
myblue2_hex = u'#64b5cd'
myred2_hex = '#ff9f9a'


def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap

cmap = plt.get_cmap('nipy_spectral')
new_cmap = truncate_colormap(cmap, 0.2, 0.95)

#colors1 = plt.cm.YlGnBu(np.linspace(0, 1, 128))
first = int((128*2)-np.round(255*(1.-0.60)))
second = (256-first)
#colors2 = new_cmap(np.linspace(0, 1, first))
colors2 = plt.cm.viridis(np.linspace(0.1, .98, first))
colors3 = plt.cm.YlOrBr(np.linspace(0.25, 1, second))
colors4 = plt.cm.PuBu(np.linspace(0., 0.5, second))
#colors4 = plt.cm.pink(np.linspace(0.9, 1., second))
# combine them and build a new colormap
cols = np.vstack((colors2,colors3))
mymap = colors.LinearSegmentedColormap.from_list('my_colormap', cols)

num = 256
gradient = range(num)
for x in range(5):
    gradient = np.vstack((gradient, gradient))

fig, ax = plt.subplots(nrows=1)
ax.imshow(gradient, cmap=mymap, interpolation='nearest')
ax.set_axis_off()
fig.tight_layout()

plt.show()