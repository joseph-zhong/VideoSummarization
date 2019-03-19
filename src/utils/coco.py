"""
coco.py
---

Utilities for interfacing with the COCO dataset.

Reference:
    - https://github.com/Yugnaynehc/coco-caption/
    -

"""
import sys
import json
import hashlib

class CocoAnnotations:

    def __init__(self):
        self.images = []
        self.annotations = []
        self.img_dict = {}
        info = {
            "year": 2017,
            "version": '1',
            "description": 'Video CaptionEval',
            "contributor": 'Subhashini Venugopalan, Yangyu Chen',
            "url": 'https://github.com/vsubhashini/, https://github.com/Yugnaynehc/',
            "date_created": '',
        }
        licenses = [{"id": 1, "name": "test", "url": "test"}]
        self.res = {"info": info,
                    "type": 'captions',
                    "images": self.images,
                    "annotations": self.annotations,
                    "licenses": licenses,
                    }

    def read_multiple_files(self, filelist):
        for filename in filelist:
            print('In file %s\n' % filename)
            self.read_file(filename)

    def get_image_dict(self, img_name):
        code = img_name.encode('utf8')
        image_hash = int(int(hashlib.sha256(code).hexdigest(), 16) % sys.maxsize)
        if image_hash in self.img_dict:
            assert self.img_dict[image_hash] == img_name, 'hash colision: {0}: {1}'.format(
                image_hash, img_name)
        else:
            self.img_dict[image_hash] = img_name
        image_dict = {"id": image_hash,
                      "width": 0,
                      "height": 0,
                      "file_name": img_name,
                      "license": '',
                      "url": img_name,
                      "date_captured": '',
                      }
        return image_dict, image_hash

    def read_file(self, filename):
        count = 0
        with open(filename, 'r') as opfd:
            for line in opfd:
                count += 1
                id_sent = line.strip().split('\t')
                try:
                    assert len(id_sent) == 2
                    sent = id_sent[1]
                except Exception as e:
                    # print(line)
                    continue
                image_dict, image_hash = self.get_image_dict(id_sent[0])
                self.images.append(image_dict)

                self.annotations.append({
                    "id": len(self.annotations) + 1,
                    "image_id": image_hash,
                    "caption": sent,
                })

    def dump_json(self, outfile):
        self.res["images"] = self.images
        self.res["annotations"] = self.annotations
        with open(outfile, 'w') as fd:
            json.dump(self.res, fd, ensure_ascii=False, sort_keys=True,
                      indent=2, separators=(',', ': '))


class CocoResFormat:

    def __init__(self):
        self.res = []
        self.caption_dict = {}

    def read_multiple_files(self, filelist, hash_img_name):
        for filename in filelist:
            print('In file %s\n' % filename)
            self.read_file(filename, hash_img_name)

    def read_file(self, filename, hash_img_name):
        count = 0
        with open(filename, 'r') as opfd:
            for line in opfd:
                count += 1
                id_sent = line.strip().split('\t')
                if len(id_sent) > 2:
                    id_sent = id_sent[-2:]
                assert len(id_sent) == 2
                sent = id_sent[1]

                if hash_img_name:
                    img_id = int(int(hashlib.sha256(id_sent[0].encode('utf8')).hexdigest(),
                                     16) % sys.maxsize)
                else:
                    img = id_sent[0].split('_')[-1].split('.')[0]
                    img_id = int(img)
                imgid_sent = {}

                if img_id in self.caption_dict:
                    assert self.caption_dict[img_id] == sent
                else:
                    self.caption_dict[img_id] = sent
                    imgid_sent['image_id'] = img_id
                    imgid_sent['caption'] = sent
                    self.res.append(imgid_sent)

    def dump_json(self, outfile):
        with open(outfile, 'w') as fd:
            json.dump(self.res, fd, ensure_ascii=False, sort_keys=True, indent=2, separators=(',', ': '))


def create_reference_json(reference_txt_path):
    crf = CocoAnnotations()
    crf.read_file(reference_txt_path)
    crf.dump_json(reference_txt_path)
    print("Created json references in {}".format(reference_txt_path))
