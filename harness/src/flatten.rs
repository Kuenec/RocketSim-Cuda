use std::collections::BTreeMap;
use std::fmt;

use serde::Serialize;
use serde::ser::{self, Impossible};

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Leaf {
    F32(u32),
    F64(u64),
    Int(i128),
    Bool(bool),
    Str(String),
    None,
    Unit,
}

impl fmt::Display for Leaf {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Leaf::F32(bits) => write!(f, "{:e} (0x{bits:08x})", f32::from_bits(*bits)),
            Leaf::F64(bits) => write!(f, "{:e} (0x{bits:016x})", f64::from_bits(*bits)),
            Leaf::Int(v) => write!(f, "{v}"),
            Leaf::Bool(v) => write!(f, "{v}"),
            Leaf::Str(s) => write!(f, "{s}"),
            Leaf::None => write!(f, "None"),
            Leaf::Unit => write!(f, "()"),
        }
    }
}

pub type Flat = BTreeMap<String, Leaf>;

#[derive(Debug)]
pub struct FlattenError(String);

impl fmt::Display for FlattenError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for FlattenError {}

impl ser::Error for FlattenError {
    fn custom<T: fmt::Display>(msg: T) -> Self {
        FlattenError(msg.to_string())
    }
}

pub fn flatten<T: Serialize>(value: &T) -> Flat {
    let mut out = Flat::new();
    value
        .serialize(Flattener {
            out: &mut out,
            path: String::new(),
        })
        .expect("flattening a state never fails");
    out
}

struct Flattener<'a> {
    out: &'a mut Flat,
    path: String,
}

impl<'a> Flattener<'a> {
    fn child(&mut self, segment: &str) -> Flattener<'_> {
        let path = if self.path.is_empty() {
            segment.to_string()
        } else {
            format!("{}.{segment}", self.path)
        };
        Flattener {
            out: self.out,
            path,
        }
    }

    fn leaf(self, leaf: Leaf) -> Result<(), FlattenError> {
        self.out.insert(self.path, leaf);
        Ok(())
    }
}

pub struct Compound<'a> {
    out: &'a mut Flat,
    path: String,
    index: usize,
}

impl Compound<'_> {
    fn next(&mut self) -> Flattener<'_> {
        let segment = self.index.to_string();
        self.index += 1;
        let path = if self.path.is_empty() {
            segment
        } else {
            format!("{}.{segment}", self.path)
        };
        Flattener {
            out: self.out,
            path,
        }
    }

    fn named(&mut self, key: &str) -> Flattener<'_> {
        let path = if self.path.is_empty() {
            key.to_string()
        } else {
            format!("{}.{key}", self.path)
        };
        Flattener {
            out: self.out,
            path,
        }
    }
}

impl<'a> ser::Serializer for Flattener<'a> {
    type Ok = ();
    type Error = FlattenError;
    type SerializeSeq = Compound<'a>;
    type SerializeTuple = Compound<'a>;
    type SerializeTupleStruct = Compound<'a>;
    type SerializeTupleVariant = Compound<'a>;
    type SerializeMap = Impossible<(), FlattenError>;
    type SerializeStruct = Compound<'a>;
    type SerializeStructVariant = Compound<'a>;

    fn serialize_bool(self, v: bool) -> Result<(), FlattenError> {
        self.leaf(Leaf::Bool(v))
    }
    fn serialize_i8(self, v: i8) -> Result<(), FlattenError> {
        self.leaf(Leaf::Int(v.into()))
    }
    fn serialize_i16(self, v: i16) -> Result<(), FlattenError> {
        self.leaf(Leaf::Int(v.into()))
    }
    fn serialize_i32(self, v: i32) -> Result<(), FlattenError> {
        self.leaf(Leaf::Int(v.into()))
    }
    fn serialize_i64(self, v: i64) -> Result<(), FlattenError> {
        self.leaf(Leaf::Int(v.into()))
    }
    fn serialize_u8(self, v: u8) -> Result<(), FlattenError> {
        self.leaf(Leaf::Int(v.into()))
    }
    fn serialize_u16(self, v: u16) -> Result<(), FlattenError> {
        self.leaf(Leaf::Int(v.into()))
    }
    fn serialize_u32(self, v: u32) -> Result<(), FlattenError> {
        self.leaf(Leaf::Int(v.into()))
    }
    fn serialize_u64(self, v: u64) -> Result<(), FlattenError> {
        self.leaf(Leaf::Int(v.into()))
    }
    fn serialize_f32(self, v: f32) -> Result<(), FlattenError> {
        self.leaf(Leaf::F32(v.to_bits()))
    }
    fn serialize_f64(self, v: f64) -> Result<(), FlattenError> {
        self.leaf(Leaf::F64(v.to_bits()))
    }
    fn serialize_char(self, v: char) -> Result<(), FlattenError> {
        self.leaf(Leaf::Str(v.to_string()))
    }
    fn serialize_str(self, v: &str) -> Result<(), FlattenError> {
        self.leaf(Leaf::Str(v.to_string()))
    }
    fn serialize_bytes(self, v: &[u8]) -> Result<(), FlattenError> {
        self.leaf(Leaf::Str(format!("{v:?}")))
    }
    fn serialize_none(self) -> Result<(), FlattenError> {
        self.leaf(Leaf::None)
    }
    fn serialize_some<T: ?Sized + Serialize>(mut self, value: &T) -> Result<(), FlattenError> {
        value.serialize(self.child("some"))
    }
    fn serialize_unit(self) -> Result<(), FlattenError> {
        self.leaf(Leaf::Unit)
    }
    fn serialize_unit_struct(self, _name: &'static str) -> Result<(), FlattenError> {
        self.leaf(Leaf::Unit)
    }
    fn serialize_unit_variant(
        self,
        _name: &'static str,
        _index: u32,
        variant: &'static str,
    ) -> Result<(), FlattenError> {
        self.leaf(Leaf::Str(variant.to_string()))
    }
    fn serialize_newtype_struct<T: ?Sized + Serialize>(
        self,
        _name: &'static str,
        value: &T,
    ) -> Result<(), FlattenError> {
        value.serialize(self)
    }
    fn serialize_newtype_variant<T: ?Sized + Serialize>(
        mut self,
        _name: &'static str,
        _index: u32,
        variant: &'static str,
        value: &T,
    ) -> Result<(), FlattenError> {
        value.serialize(self.child(variant))
    }
    fn serialize_seq(self, _len: Option<usize>) -> Result<Compound<'a>, FlattenError> {
        Ok(Compound {
            out: self.out,
            path: self.path,
            index: 0,
        })
    }
    fn serialize_tuple(self, _len: usize) -> Result<Compound<'a>, FlattenError> {
        Ok(Compound {
            out: self.out,
            path: self.path,
            index: 0,
        })
    }
    fn serialize_tuple_struct(
        self,
        _name: &'static str,
        _len: usize,
    ) -> Result<Compound<'a>, FlattenError> {
        Ok(Compound {
            out: self.out,
            path: self.path,
            index: 0,
        })
    }
    fn serialize_tuple_variant(
        mut self,
        _name: &'static str,
        _index: u32,
        variant: &'static str,
        _len: usize,
    ) -> Result<Compound<'a>, FlattenError> {
        let path = self.child(variant).path;
        Ok(Compound {
            out: self.out,
            path,
            index: 0,
        })
    }
    fn serialize_map(self, _len: Option<usize>) -> Result<Self::SerializeMap, FlattenError> {
        Err(FlattenError(
            "maps are not part of the compared state".into(),
        ))
    }
    fn serialize_struct(
        self,
        _name: &'static str,
        _len: usize,
    ) -> Result<Compound<'a>, FlattenError> {
        Ok(Compound {
            out: self.out,
            path: self.path,
            index: 0,
        })
    }
    fn serialize_struct_variant(
        mut self,
        _name: &'static str,
        _index: u32,
        variant: &'static str,
        _len: usize,
    ) -> Result<Compound<'a>, FlattenError> {
        let path = self.child(variant).path;
        Ok(Compound {
            out: self.out,
            path,
            index: 0,
        })
    }
}

impl ser::SerializeSeq for Compound<'_> {
    type Ok = ();
    type Error = FlattenError;
    fn serialize_element<T: ?Sized + Serialize>(&mut self, value: &T) -> Result<(), FlattenError> {
        value.serialize(self.next())
    }
    fn end(self) -> Result<(), FlattenError> {
        Ok(())
    }
}

impl ser::SerializeTuple for Compound<'_> {
    type Ok = ();
    type Error = FlattenError;
    fn serialize_element<T: ?Sized + Serialize>(&mut self, value: &T) -> Result<(), FlattenError> {
        value.serialize(self.next())
    }
    fn end(self) -> Result<(), FlattenError> {
        Ok(())
    }
}

impl ser::SerializeTupleStruct for Compound<'_> {
    type Ok = ();
    type Error = FlattenError;
    fn serialize_field<T: ?Sized + Serialize>(&mut self, value: &T) -> Result<(), FlattenError> {
        value.serialize(self.next())
    }
    fn end(self) -> Result<(), FlattenError> {
        Ok(())
    }
}

impl ser::SerializeTupleVariant for Compound<'_> {
    type Ok = ();
    type Error = FlattenError;
    fn serialize_field<T: ?Sized + Serialize>(&mut self, value: &T) -> Result<(), FlattenError> {
        value.serialize(self.next())
    }
    fn end(self) -> Result<(), FlattenError> {
        Ok(())
    }
}

impl ser::SerializeStruct for Compound<'_> {
    type Ok = ();
    type Error = FlattenError;
    fn serialize_field<T: ?Sized + Serialize>(
        &mut self,
        key: &'static str,
        value: &T,
    ) -> Result<(), FlattenError> {
        value.serialize(self.named(key))
    }
    fn end(self) -> Result<(), FlattenError> {
        Ok(())
    }
}

impl ser::SerializeStructVariant for Compound<'_> {
    type Ok = ();
    type Error = FlattenError;
    fn serialize_field<T: ?Sized + Serialize>(
        &mut self,
        key: &'static str,
        value: &T,
    ) -> Result<(), FlattenError> {
        value.serialize(self.named(key))
    }
    fn end(self) -> Result<(), FlattenError> {
        Ok(())
    }
}

pub fn first_difference(expected: &Flat, actual: &Flat) -> Option<String> {
    for (key, e) in expected {
        match actual.get(key) {
            None => {
                return Some(format!(
                    "{key}: expected {e}, missing in GPU/CPU port state"
                ));
            }
            Some(a) if a != e => return Some(format!("{key}: expected {e}, got {a}")),
            Some(_) => {}
        }
    }
    for (key, a) in actual {
        if !expected.contains_key(key) {
            return Some(format!("{key}: unexpected field with value {a}"));
        }
    }
    None
}
