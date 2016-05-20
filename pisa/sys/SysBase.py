class SysBase(object):

    def get_scales(self,sys_val):
        return 1.0

    def get_hi_scales(self,hi_val, hifwd_val):
        return 1.0

    def apply_sys(self, maps, sys_val):
        output_maps = {}
        output_maps['params'] = maps['params']
        for channel in ['trck', 'cscd']:
            output_maps[channel] = { 'map': (maps[channel]['map']) * self.get_scales(channel, sys_val),
                                     'ebins':maps[channel]['ebins'],
                                     'czbins': maps[channel]['czbins']}
        return output_maps

    def apply_hi_hifwd(self, maps, hi_val, hifwd_val):
        output_maps = {}
        output_maps['params'] = maps['params']
        for channel in ['trck', 'cscd']:
            output_maps[channel] = { 'map': (maps[channel]['map']) * self.get_hi_scales(channel, hi_val, hifwd_val),
                                     'ebins':maps[channel]['ebins'],
                                     'czbins': maps[channel]['czbins']}
        return output_maps
