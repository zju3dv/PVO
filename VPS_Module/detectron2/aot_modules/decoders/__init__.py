from detectron2.aot_modules.decoders.fpn import FPNSegmentationHead


def build_decoder(name, **kwargs):

    if name == 'fpn':
        return FPNSegmentationHead(**kwargs)
    else:
        raise NotImplementedError
