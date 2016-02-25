class SysBase(object):

    def get_scales(self,sys_val):
        return 1.0

    def apply_sys(self, maps, sys_val):
        output_maps = {}
        for channel in ['trck', 'cscd']:
            output_maps[channel] = { 'map': (maps[channel]['map']) * self.get_scales(channel, sys_val),
                                     'ebins':maps[channel]['ebins'],
                                     'czbins': maps[channel]['czbins'] }
        return output_maps
