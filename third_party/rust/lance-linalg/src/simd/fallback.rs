// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::ops::{Add, AddAssign, Mul, Sub, SubAssign};
use super::{FloatSimd, SIMD};

// --- f32x8 ---
#[derive(Clone, Copy)]
pub struct f32x8(pub [f32; 8]);

impl std::fmt::Debug for f32x8 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "f32x8({:?})", self.0)
    }
}

impl f32x8 {
    #[inline]
    pub fn gather(slice: &[f32], indices: &[i32; 8]) -> Self {
        let ptr = slice.as_ptr();
        unsafe {
            Self([
                *ptr.add(indices[0] as usize),
                *ptr.add(indices[1] as usize),
                *ptr.add(indices[2] as usize),
                *ptr.add(indices[3] as usize),
                *ptr.add(indices[4] as usize),
                *ptr.add(indices[5] as usize),
                *ptr.add(indices[6] as usize),
                *ptr.add(indices[7] as usize),
            ])
        }
    }
}

impl From<&[f32]> for f32x8 {
    #[inline]
    fn from(value: &[f32]) -> Self {
        unsafe { Self::load_unaligned(value.as_ptr()) }
    }
}

impl<'a> From<&'a [f32; 8]> for f32x8 {
    #[inline]
    fn from(value: &'a [f32; 8]) -> Self {
        unsafe { Self::load_unaligned(value.as_ptr()) }
    }
}

impl SIMD<f32, 8> for f32x8 {
    #[inline]
    fn splat(val: f32) -> Self {
        Self([val; 8])
    }

    #[inline]
    fn zeros() -> Self {
        Self([0.0; 8])
    }

    #[inline]
    unsafe fn load(ptr: *const f32) -> Self {
        Self::load_unaligned(ptr)
    }

    #[inline]
    unsafe fn load_unaligned(ptr: *const f32) -> Self {
        let mut arr = [0.0; 8];
        std::ptr::copy_nonoverlapping(ptr, arr.as_mut_ptr(), 8);
        Self(arr)
    }

    #[inline]
    unsafe fn store(&self, ptr: *mut f32) {
        self.store_unaligned(ptr);
    }

    #[inline]
    unsafe fn store_unaligned(&self, ptr: *mut f32) {
        std::ptr::copy_nonoverlapping(self.0.as_ptr(), ptr, 8);
    }

    #[inline]
    fn reduce_sum(&self) -> f32 {
        self.0.iter().sum()
    }

    #[inline]
    fn reduce_min(&self) -> f32 {
        self.0.iter().copied().fold(f32::INFINITY, f32::min)
    }

    #[inline]
    fn min(&self, rhs: &Self) -> Self {
        let mut arr = [0.0; 8];
        for i in 0..8 {
            arr[i] = self.0[i].min(rhs.0[i]);
        }
        Self(arr)
    }

    #[inline]
    fn find(&self, val: f32) -> Option<i32> {
        self.0.iter().position(|&x| x == val).map(|idx| idx as i32)
    }
}

impl FloatSimd<f32, 8> for f32x8 {
    #[inline]
    fn multiply_add(&mut self, a: Self, b: Self) {
        for i in 0..8 {
            self.0[i] += a.0[i] * b.0[i];
        }
    }
}

impl Add for f32x8 {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self {
        let mut arr = [0.0; 8];
        for i in 0..8 {
            arr[i] = self.0[i] + rhs.0[i];
        }
        Self(arr)
    }
}
impl AddAssign for f32x8 {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        for i in 0..8 {
            self.0[i] += rhs.0[i];
        }
    }
}
impl Sub for f32x8 {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self {
        let mut arr = [0.0; 8];
        for i in 0..8 {
            arr[i] = self.0[i] - rhs.0[i];
        }
        Self(arr)
    }
}
impl SubAssign for f32x8 {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        for i in 0..8 {
            self.0[i] -= rhs.0[i];
        }
    }
}
impl Mul for f32x8 {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: Self) -> Self {
        let mut arr = [0.0; 8];
        for i in 0..8 {
            arr[i] = self.0[i] * rhs.0[i];
        }
        Self(arr)
    }
}

// --- f32x16 ---
#[derive(Clone, Copy)]
pub struct f32x16(pub [f32; 16]);

impl std::fmt::Debug for f32x16 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "f32x16({:?})", self.0)
    }
}

impl From<&[f32]> for f32x16 {
    #[inline]
    fn from(value: &[f32]) -> Self {
        unsafe { Self::load_unaligned(value.as_ptr()) }
    }
}

impl<'a> From<&'a [f32; 16]> for f32x16 {
    #[inline]
    fn from(value: &'a [f32; 16]) -> Self {
        unsafe { Self::load_unaligned(value.as_ptr()) }
    }
}

impl SIMD<f32, 16> for f32x16 {
    #[inline]
    fn splat(val: f32) -> Self {
        Self([val; 16])
    }

    #[inline]
    fn zeros() -> Self {
        Self([0.0; 16])
    }

    #[inline]
    unsafe fn load(ptr: *const f32) -> Self {
        Self::load_unaligned(ptr)
    }

    #[inline]
    unsafe fn load_unaligned(ptr: *const f32) -> Self {
        let mut arr = [0.0; 16];
        std::ptr::copy_nonoverlapping(ptr, arr.as_mut_ptr(), 16);
        Self(arr)
    }

    #[inline]
    unsafe fn store(&self, ptr: *mut f32) {
        self.store_unaligned(ptr);
    }

    #[inline]
    unsafe fn store_unaligned(&self, ptr: *mut f32) {
        std::ptr::copy_nonoverlapping(self.0.as_ptr(), ptr, 16);
    }

    #[inline]
    fn reduce_sum(&self) -> f32 {
        self.0.iter().sum()
    }

    #[inline]
    fn reduce_min(&self) -> f32 {
        self.0.iter().copied().fold(f32::INFINITY, f32::min)
    }

    #[inline]
    fn min(&self, rhs: &Self) -> Self {
        let mut arr = [0.0; 16];
        for i in 0..16 {
            arr[i] = self.0[i].min(rhs.0[i]);
        }
        Self(arr)
    }

    #[inline]
    fn find(&self, val: f32) -> Option<i32> {
        self.0.iter().position(|&x| x == val).map(|idx| idx as i32)
    }
}

impl FloatSimd<f32, 16> for f32x16 {
    #[inline]
    fn multiply_add(&mut self, a: Self, b: Self) {
        for i in 0..16 {
            self.0[i] += a.0[i] * b.0[i];
        }
    }
}

impl Add for f32x16 {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self {
        let mut arr = [0.0; 16];
        for i in 0..16 {
            arr[i] = self.0[i] + rhs.0[i];
        }
        Self(arr)
    }
}
impl AddAssign for f32x16 {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        for i in 0..16 {
            self.0[i] += rhs.0[i];
        }
    }
}
impl Sub for f32x16 {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self {
        let mut arr = [0.0; 16];
        for i in 0..16 {
            arr[i] = self.0[i] - rhs.0[i];
        }
        Self(arr)
    }
}
impl SubAssign for f32x16 {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        for i in 0..16 {
            self.0[i] -= rhs.0[i];
        }
    }
}
impl Mul for f32x16 {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: Self) -> Self {
        let mut arr = [0.0; 16];
        for i in 0..16 {
            arr[i] = self.0[i] * rhs.0[i];
        }
        Self(arr)
    }
}

// --- f64x4 ---
#[derive(Clone, Copy)]
pub struct f64x4(pub [f64; 4]);

impl std::fmt::Debug for f64x4 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "f64x4({:?})", self.0)
    }
}

impl From<&[f64]> for f64x4 {
    #[inline]
    fn from(value: &[f64]) -> Self {
        unsafe { Self::load_unaligned(value.as_ptr()) }
    }
}

impl<'a> From<&'a [f64; 4]> for f64x4 {
    #[inline]
    fn from(value: &'a [f64; 4]) -> Self {
        unsafe { Self::load_unaligned(value.as_ptr()) }
    }
}

impl SIMD<f64, 4> for f64x4 {
    #[inline]
    fn splat(val: f64) -> Self {
        Self([val; 4])
    }

    #[inline]
    fn zeros() -> Self {
        Self([0.0; 4])
    }

    #[inline]
    unsafe fn load(ptr: *const f64) -> Self {
        Self::load_unaligned(ptr)
    }

    #[inline]
    unsafe fn load_unaligned(ptr: *const f64) -> Self {
        let mut arr = [0.0; 4];
        std::ptr::copy_nonoverlapping(ptr, arr.as_mut_ptr(), 4);
        Self(arr)
    }

    #[inline]
    unsafe fn store(&self, ptr: *mut f64) {
        self.store_unaligned(ptr);
    }

    #[inline]
    unsafe fn store_unaligned(&self, ptr: *mut f64) {
        std::ptr::copy_nonoverlapping(self.0.as_ptr(), ptr, 4);
    }

    #[inline]
    fn reduce_sum(&self) -> f64 {
        self.0.iter().sum()
    }

    #[inline]
    fn reduce_min(&self) -> f64 {
        self.0.iter().copied().fold(f64::INFINITY, f64::min)
    }

    #[inline]
    fn min(&self, rhs: &Self) -> Self {
        let mut arr = [0.0; 4];
        for i in 0..4 {
            arr[i] = self.0[i].min(rhs.0[i]);
        }
        Self(arr)
    }

    #[inline]
    fn find(&self, val: f64) -> Option<i32> {
        self.0.iter().position(|&x| x == val).map(|idx| idx as i32)
    }
}

impl FloatSimd<f64, 4> for f64x4 {
    #[inline]
    fn multiply_add(&mut self, a: Self, b: Self) {
        for i in 0..4 {
            self.0[i] += a.0[i] * b.0[i];
        }
    }
}

impl Add for f64x4 {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self {
        let mut arr = [0.0; 4];
        for i in 0..4 {
            arr[i] = self.0[i] + rhs.0[i];
        }
        Self(arr)
    }
}
impl AddAssign for f64x4 {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        for i in 0..4 {
            self.0[i] += rhs.0[i];
        }
    }
}
impl Sub for f64x4 {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self {
        let mut arr = [0.0; 4];
        for i in 0..4 {
            arr[i] = self.0[i] - rhs.0[i];
        }
        Self(arr)
    }
}
impl SubAssign for f64x4 {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        for i in 0..4 {
            self.0[i] -= rhs.0[i];
        }
    }
}
impl Mul for f64x4 {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: Self) -> Self {
        let mut arr = [0.0; 4];
        for i in 0..4 {
            arr[i] = self.0[i] * rhs.0[i];
        }
        Self(arr)
    }
}

// --- f64x8 ---
#[derive(Clone, Copy)]
pub struct f64x8(pub [f64; 8]);

impl std::fmt::Debug for f64x8 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "f64x8({:?})", self.0)
    }
}

impl From<&[f64]> for f64x8 {
    #[inline]
    fn from(value: &[f64]) -> Self {
        unsafe { Self::load_unaligned(value.as_ptr()) }
    }
}

impl<'a> From<&'a [f64; 8]> for f64x8 {
    #[inline]
    fn from(value: &'a [f64; 8]) -> Self {
        unsafe { Self::load_unaligned(value.as_ptr()) }
    }
}

impl SIMD<f64, 8> for f64x8 {
    #[inline]
    fn splat(val: f64) -> Self {
        Self([val; 8])
    }

    #[inline]
    fn zeros() -> Self {
        Self([0.0; 8])
    }

    #[inline]
    unsafe fn load(ptr: *const f64) -> Self {
        Self::load_unaligned(ptr)
    }

    #[inline]
    unsafe fn load_unaligned(ptr: *const f64) -> Self {
        let mut arr = [0.0; 8];
        std::ptr::copy_nonoverlapping(ptr, arr.as_mut_ptr(), 8);
        Self(arr)
    }

    #[inline]
    unsafe fn store(&self, ptr: *mut f64) {
        self.store_unaligned(ptr);
    }

    #[inline]
    unsafe fn store_unaligned(&self, ptr: *mut f64) {
        std::ptr::copy_nonoverlapping(self.0.as_ptr(), ptr, 8);
    }

    #[inline]
    fn reduce_sum(&self) -> f64 {
        self.0.iter().sum()
    }

    #[inline]
    fn reduce_min(&self) -> f64 {
        self.0.iter().copied().fold(f64::INFINITY, f64::min)
    }

    #[inline]
    fn min(&self, rhs: &Self) -> Self {
        let mut arr = [0.0; 8];
        for i in 0..8 {
            arr[i] = self.0[i].min(rhs.0[i]);
        }
        Self(arr)
    }

    #[inline]
    fn find(&self, val: f64) -> Option<i32> {
        self.0.iter().position(|&x| x == val).map(|idx| idx as i32)
    }
}

impl FloatSimd<f64, 8> for f64x8 {
    #[inline]
    fn multiply_add(&mut self, a: Self, b: Self) {
        for i in 0..8 {
            self.0[i] += a.0[i] * b.0[i];
        }
    }
}

impl Add for f64x8 {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self {
        let mut arr = [0.0; 8];
        for i in 0..8 {
            arr[i] = self.0[i] + rhs.0[i];
        }
        Self(arr)
    }
}
impl AddAssign for f64x8 {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        for i in 0..8 {
            self.0[i] += rhs.0[i];
        }
    }
}
impl Sub for f64x8 {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self {
        let mut arr = [0.0; 8];
        for i in 0..8 {
            arr[i] = self.0[i] - rhs.0[i];
        }
        Self(arr)
    }
}
impl SubAssign for f64x8 {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        for i in 0..8 {
            self.0[i] -= rhs.0[i];
        }
    }
}
impl Mul for f64x8 {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: Self) -> Self {
        let mut arr = [0.0; 8];
        for i in 0..8 {
            arr[i] = self.0[i] * rhs.0[i];
        }
        Self(arr)
    }
}

// --- i32x8 ---
#[derive(Clone, Copy)]
pub struct i32x8(pub [i32; 8]);

impl std::fmt::Debug for i32x8 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "i32x8({:?})", self.0)
    }
}

impl From<&[i32]> for i32x8 {
    #[inline]
    fn from(value: &[i32]) -> Self {
        unsafe { Self::load_unaligned(value.as_ptr()) }
    }
}

impl From<&[i32; 8]> for i32x8 {
    #[inline]
    fn from(value: &[i32; 8]) -> Self {
        unsafe { Self::load_unaligned(value.as_ptr()) }
    }
}

impl SIMD<i32, 8> for i32x8 {
    #[inline]
    fn splat(val: i32) -> Self {
        Self([val; 8])
    }

    #[inline]
    fn zeros() -> Self {
        Self([0; 8])
    }

    #[inline]
    unsafe fn load(ptr: *const i32) -> Self {
        Self::load_unaligned(ptr)
    }

    #[inline]
    unsafe fn load_unaligned(ptr: *const i32) -> Self {
        let mut arr = [0; 8];
        std::ptr::copy_nonoverlapping(ptr, arr.as_mut_ptr(), 8);
        Self(arr)
    }

    #[inline]
    unsafe fn store(&self, ptr: *mut i32) {
        self.store_unaligned(ptr);
    }

    #[inline]
    unsafe fn store_unaligned(&self, ptr: *mut i32) {
        std::ptr::copy_nonoverlapping(self.0.as_ptr(), ptr, 8);
    }

    #[inline]
    fn reduce_sum(&self) -> i32 {
        self.0.iter().sum()
    }

    #[inline]
    fn reduce_min(&self) -> i32 {
        self.0.iter().copied().fold(i32::MAX, i32::min)
    }

    #[inline]
    fn min(&self, rhs: &Self) -> Self {
        let mut arr = [0; 8];
        for i in 0..8 {
            arr[i] = self.0[i].min(rhs.0[i]);
        }
        Self(arr)
    }

    #[inline]
    fn find(&self, val: i32) -> Option<i32> {
        self.0.iter().position(|&x| x == val).map(|idx| idx as i32)
    }
}

impl Add for i32x8 {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self {
        let mut arr = [0; 8];
        for i in 0..8 {
            arr[i] = self.0[i] + rhs.0[i];
        }
        Self(arr)
    }
}
impl AddAssign for i32x8 {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        for i in 0..8 {
            self.0[i] += rhs.0[i];
        }
    }
}
impl Sub for i32x8 {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self {
        let mut arr = [0; 8];
        for i in 0..8 {
            arr[i] = self.0[i] - rhs.0[i];
        }
        Self(arr)
    }
}
impl SubAssign for i32x8 {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        for i in 0..8 {
            self.0[i] -= rhs.0[i];
        }
    }
}
impl Mul for i32x8 {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: Self) -> Self {
        let mut arr = [0; 8];
        for i in 0..8 {
            arr[i] = self.0[i] * rhs.0[i];
        }
        Self(arr)
    }
}
