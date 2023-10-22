# Import the libraries
# These lines of code are importing specific classes and functions from the Bokeh library.
from bokeh.plotting import figure, show
from bokeh.models import ColumnDataSource
from bokeh.transform import factor_cmap
from bokeh.models import HoverTool

# create_bokeh_plot function
def create_bokeh_plot(review_counts):
		"""
		The function `create_bokeh_plot` creates a Bokeh plot with a bar chart of review categories and
		their corresponding counts.
		
		:param review_counts: The `review_counts` parameter is a dictionary that contains the categories of
		reviews as keys and the corresponding counts of reviews as values
		:return: a Bokeh plot object.
		"""
		categories = list(review_counts.keys())
		counts = list(review_counts.values())

		# The line `source = ColumnDataSource(data=dict(categories=categories, counts=counts))` is creating
		# a ColumnDataSource object.
		source = ColumnDataSource(data=dict(categories=categories, counts=counts))

		p = figure(x_range=categories, height=350, title="Review Categories", toolbar_location=None, tools="")

		# Add hover tool
		# The code block is creating a HoverTool object and configuring its tooltips.
		hover = HoverTool()
		hover.tooltips = [("Category", "@categories"), ("Count", "@counts")]
		p.add_tools(hover)

		# The line `p.vbar(x='categories', top='counts', width=0.5, source=source, line_color="white",
		# fill_color=factor_cmap('categories', palette=['#FF5733', '#3399FF', '#33FF57', '#FF33A1'],
		# factors=categories))` is creating a vertical bar chart in the Bokeh plot.
		p.vbar(x='categories', top='counts', width=0.5, source=source, line_color="white",
			fill_color=factor_cmap('categories', palette=['#FF5733', '#3399FF', '#33FF57', '#FF33A1'], factors=categories))

		# These lines of code are configuring various properties of the Bokeh plot object `p`:

		# The line `p.xgrid.grid_line_color = None` is setting the color of the grid lines along the x-axis
		# to None. This means that the grid lines will not be visible in the plot.
		p.xgrid.grid_line_color = None

		# The line `p.y_range.start = 0` is setting the starting value of the y-axis range to 0. This means
		# that the y-axis will start from 0 on the plot.
		p.y_range.start = 0

		# The line `p.y_range.end = max(counts) + 5` is setting the end value of the y-axis range in the
		# Bokeh plot.
		p.y_range.end = max(counts) + 5

		# The line `p.title.text_font_size = "16pt"` is setting the font size of the title of the Bokeh plot
		# to 16 points.
		p.title.text_font_size = "16pt"

		# The line `p.xaxis.major_label_orientation = 1.2` is setting the orientation of the major labels on
		# the x-axis of the Bokeh plot.
		p.xaxis.major_label_orientation = 1.2

		# The line `p.yaxis.axis_label = "Number of Reviews"` is setting the label for the y-axis of the
		# Bokeh plot to "Number of Reviews". This label will be displayed on the y-axis of the plot to
		# provide a clear indication of what the values on the y-axis represent.
		p.yaxis.axis_label = "Number of Reviews"

		return p