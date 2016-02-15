from operator import mul


class MathTools():
    def __init__(self):
        pass

    def max_size(self, mat, value=0):
        """Find pos, h, w of the largest rectangle containing all `value`'s.
        For each row solve "Largest Rectangle in a Histrogram" problem [1]:
        [1]: http://blog.csdn.net/arbuckle/archive/2006/05/06/710988.aspx

        @param mat: input matrix

        Keyword arguments:
        value -- the value to be found in the rectangle

        @return (height, width), (start_y, start_x)
        """
        start_row = 0
        it = iter(mat)
        hist = [(el == value) for el in next(it, [])]
        max_size, start_pos = self.max_rectangle_size(hist)
        counter = 0
        for row in it:
            counter += 1
            hist = [(1+h) if el == value else 0 for h, el in zip(hist, row)]
            _max_size, _start = self.max_rectangle_size(hist)
            if self.area(_max_size) > self.area(max_size):
                max_size = _max_size
                start_pos = _start
                start_row = counter
        y = start_row - max_size[0] + 1
        if max_size[1] == len(hist):
            x = 0
        else:
            x = min(abs(start_pos - max_size[1] + 1), start_pos)
        return max_size, (y, x)

    def max_rectangle_size(self, histogram):
        """Find height, width of the largest rectangle that fits entirely
        under the histogram. Algorithm is "Linear search using a stack of
        incomplete subproblems" [1].
        [1]: http://blog.csdn.net/arbuckle/archive/2006/05/06/710988.aspx
        """
        from collections import namedtuple
        Info = namedtuple('Info', 'start height')

        # Maintain a stack
        stack = []
        top = lambda: stack[-1]
        max_size = (0, 0)
        pos = 0
        for pos, height in enumerate(histogram):
            # Position where rectangle starts
            start = pos
            while True:
                # If the stack is empty, push
                if len(stack) == 0:
                    stack.append(Info(start, height))
                # If the right bar is higher than the current top, push
                elif height > top().height:
                    stack.append(Info(start, height))
                # Else, calculate the rectangle size
                elif stack and height < top().height:
                    max_size = max(max_size, (top().height,
                                   (pos - top().start)), key=self.area)
                    start, _ = stack.pop()
                    continue
                # Height == top().height goes here
                break

        pos += 1
        start_pos = 0
        for start, height in stack:
            _max_size = max(max_size, (height, (pos - start)), key=self.area)
            if self.area(_max_size) >= self.area(max_size):
                max_size = _max_size
                start_pos = start

        return max_size, start_pos

    def area(self, size):
        return reduce(mul, size)


