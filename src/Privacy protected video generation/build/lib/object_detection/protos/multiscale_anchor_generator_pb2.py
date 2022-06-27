# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: object_detection/protos/multiscale_anchor_generator.proto

import sys
_b=sys.version_info[0]<3 and (lambda x:x) or (lambda x:x.encode('latin1'))
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor.FileDescriptor(
  name='object_detection/protos/multiscale_anchor_generator.proto',
  package='object_detection.protos',
  syntax='proto2',
  serialized_options=None,
  serialized_pb=_b('\n9object_detection/protos/multiscale_anchor_generator.proto\x12\x17object_detection.protos\"\xba\x01\n\x19MultiscaleAnchorGenerator\x12\x14\n\tmin_level\x18\x01 \x01(\x05:\x01\x33\x12\x14\n\tmax_level\x18\x02 \x01(\x05:\x01\x37\x12\x17\n\x0c\x61nchor_scale\x18\x03 \x01(\x02:\x01\x34\x12\x15\n\raspect_ratios\x18\x04 \x03(\x02\x12\x1c\n\x11scales_per_octave\x18\x05 \x01(\x05:\x01\x32\x12#\n\x15normalize_coordinates\x18\x06 \x01(\x08:\x04true')
)




_MULTISCALEANCHORGENERATOR = _descriptor.Descriptor(
  name='MultiscaleAnchorGenerator',
  full_name='object_detection.protos.MultiscaleAnchorGenerator',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='min_level', full_name='object_detection.protos.MultiscaleAnchorGenerator.min_level', index=0,
      number=1, type=5, cpp_type=1, label=1,
      has_default_value=True, default_value=3,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='max_level', full_name='object_detection.protos.MultiscaleAnchorGenerator.max_level', index=1,
      number=2, type=5, cpp_type=1, label=1,
      has_default_value=True, default_value=7,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='anchor_scale', full_name='object_detection.protos.MultiscaleAnchorGenerator.anchor_scale', index=2,
      number=3, type=2, cpp_type=6, label=1,
      has_default_value=True, default_value=float(4),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='aspect_ratios', full_name='object_detection.protos.MultiscaleAnchorGenerator.aspect_ratios', index=3,
      number=4, type=2, cpp_type=6, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='scales_per_octave', full_name='object_detection.protos.MultiscaleAnchorGenerator.scales_per_octave', index=4,
      number=5, type=5, cpp_type=1, label=1,
      has_default_value=True, default_value=2,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='normalize_coordinates', full_name='object_detection.protos.MultiscaleAnchorGenerator.normalize_coordinates', index=5,
      number=6, type=8, cpp_type=7, label=1,
      has_default_value=True, default_value=True,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=87,
  serialized_end=273,
)

DESCRIPTOR.message_types_by_name['MultiscaleAnchorGenerator'] = _MULTISCALEANCHORGENERATOR
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

MultiscaleAnchorGenerator = _reflection.GeneratedProtocolMessageType('MultiscaleAnchorGenerator', (_message.Message,), {
  'DESCRIPTOR' : _MULTISCALEANCHORGENERATOR,
  '__module__' : 'object_detection.protos.multiscale_anchor_generator_pb2'
  # @@protoc_insertion_point(class_scope:object_detection.protos.MultiscaleAnchorGenerator)
  })
_sym_db.RegisterMessage(MultiscaleAnchorGenerator)


# @@protoc_insertion_point(module_scope)
