import sys
import numpy as np
import os

import matplotlib
from matplotlib.figure import Figure
from matplotlib.axes import Subplot
from matplotlib.backends.backend_gtk import FigureCanvasGTK
from matplotlib.backends.backend_gtk import NavigationToolbar2GTK as NavigationToolbar

try:
    import pygtk
    pygtk.require("2.0")
    import gtk
    import gtk.glade
except:
    sys.exit(1)

def findFileInPythonPath(fileName):
    """Find the file named path in the sys.path.
    Returns the full path name if found, None if not found"""
    for dirname in sys.path:
        possible = os.path.join(dirname, fileName)
        if os.path.isfile(possible):
            print 'loading gui from %s' % possible
            return possible
    return None

class ManipulateGUI(object):
    """woot"""

    def __init__(self,parameterDict, liveUpdate=False):
        # 'public' vars
	self.parameterDict = parameterDict
        self.liveUpdate = liveUpdate
        self.nParams = len(parameterDict)

        #Set the Glade file and build from it
        self.gladefile = findFileInPythonPath('manipulateGUI.glade')
        self.builder = gtk.Builder()
        self.builder.add_from_file(self.gladefile)

        #Grab some key gui handles
        self.window = self.builder.get_object('manipulateWindow')
        self.window.connect('destroy', gtk.main_quit)
        self.paramBox = self.builder.get_object("paramBox")

        # loop over the parameterDict and...
        # ... label for every fixed parameter
        # ... slider for every variable
        # ... dropdown for non-fixed parameter

        self.widgetList = []
        self.adjustmentList = []
        self.comboboxList = []

        for arg, params in self.parameterDict.iteritems():
            hbox = gtk.HBox(False, 0)
            label = gtk.Label(arg+':')
            label.show()
            hbox.pack_start(label, True, True)

            if params['type'] == 'fixed':
                #add a new GTKlabel
                self.widgetList.append(gtk.Label(params['value']))

            elif params['type'] == 'range':
                # a horizatonal scale bar
                # self.adjustmentList.append(gtk.Adjustment(value=params['value'], lower=params['lower'], upper=params['upper'], step_incr=params['step'], page_incr=params['page_inc'],page_size=params['page_size']))
                self.adjustmentList.append(gtk.Adjustment(value=params['value'], lower=params['lower'], upper=params['upper'], step_incr=params['step']))
                self.widgetList.append(gtk.HScale(adjustment=self.adjustmentList[-1]))

                self.widgetList[-1].connect('change-value', self.on_param_change_range_value)

            elif params['type'] == 'list':
                # build combobox from list
                self.widgetList.append(gtk.combo_box_new_text())
                for item in params['options']:
                    self.widgetList[-1].append_text(item)

                self.widgetList[-1].set_active(0)
                self.widgetList[-1].connect('changed', self.on_param_change_list_value)

            self.widgetList[-1].argName = arg

            hbox.pack_start(child=self.widgetList[-1], expand=True, fill=True)
            self.paramBox.pack_start(child=hbox, expand=True, fill=True)
            # self.paramBox.pack_start(child=self.widgetList[-1], expand=True, fill=True)


        # setup matplotlib in the matplotlibWindow
        self.figure = Figure(figsize=(10,4), dpi=72)
        self.inputAxis = self.figure.add_subplot(1,2,1)
        self.outputAxis = self.figure.add_subplot(1,2,2)
        
        self.canvas = FigureCanvasGTK(self.figure) # a gtk.DrawingArea
        self.canvas.show()
        self.canvas.draw()
        
        self.plotView = self.builder.get_object("mplBox")
        self.plotView.pack_start(self.canvas, True, True)

        # connect and init data
        self.builder.connect_signals(self)
        self.window.show_all()

    # button and checkbox callbacks
    def on_return_clicked(self, widget):
        # overload in subclass
        pass

    def on_refresh_clicked(self, widget):
        self.eval_function()

    def on_param_change_range_value(self, widget, event, paramValue):
        self.parameterDict[widget.argName]['value'] = paramValue
        if self.liveUpdate:
            self.eval_function()

    def on_param_change_list_value(self, widget):
        self.parameterDict[widget.argName]['value'] = widget.get_active()
        if self.liveUpdate:
            self.eval_function()

    def eval_function(self):
    # overload in subclass
    # should use self.parameterDict somehow... (particular for each subclass)
    # self.function(arg1 = self.parameterDict['arg1']['value'] , ... )
        pass
