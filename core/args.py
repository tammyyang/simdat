import os
from simdat.core import tools

io = tools.MLIO()

class Args(object):
    def __init__(self, pfs=['ml.json']):
        """Init function of Args

        Keyword arguments:
        pfs -- profiles to read (default: ['ml.json'])

        """
        self._add_args()
        for f in pfs:
            self._set_args(f)
        self._tune_args()

    def _add_args(self):
        """Called by __init__ of Args class"""
        pass

    def _tune_args(self):
        """Called by __init__ of Args class (after _set_args)"""
        pass

    def _set_args(self, f):
        """Read parameters from profile

        @param f: profile file

        """

        if not os.path.isfile(f):
            print("[Args] WARNING: File %s does not exist" % f)
            return
        inparm = io.parse_json(f)
        cinst = self.__dict__.keys()
        for k in inparm:
            if k in cinst:
                setattr(self, k, inparm[k])

    def _print_arg(self, arg, arg_dic):
        """Print argument explanation"""
        print('[Args] * %s *' % arg)
        print('[Args]   Description: %s' % arg_dic['des'])
        print('[Args]   Type: %s' % arg_dic['type'])
        print('[Args]   Default: %s' % arg_dic['default'])

    def explain_args(self, fname, arg=''):
        """Print explanation for args

        @param fname: The json file which includes the explanations

        Keyword Arguments:
        arg -- specify the argument to print (default: all)

        """
        intros = io.parse_json(fname)
        print('[Args] === Reading explanations from %s.' % fname)
        if len(arg) < 1:
            for _arg in intros.keys():
                self._print_arg(_arg, intros[_arg])
        else:
            self._print_arg(arg, intros[arg])
