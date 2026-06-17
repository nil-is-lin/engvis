use glam::{Vec3, Affine3A, Mat4};

#[derive(Debug, Clone, Copy)]
pub struct Aabb {
    pub min: Vec3,
    pub max: Vec3,
}

impl Aabb {
    pub fn empty() -> Self {
        Self {
            min: Vec3::splat(f32::MAX),
            max: Vec3::splat(f32::MIN),
        }
    }

    pub fn expand(&mut self, point: Vec3) {
        self.min = self.min.min(point);
        self.max = self.max.max(point);
    }

    /// Expand to include another AABB.
    pub fn union(&self, other: &Aabb) -> Aabb {
        if !other.is_valid() {
            return *self;
        }
        if !self.is_valid() {
            return *other;
        }
        Aabb {
            min: self.min.min(other.min),
            max: self.max.max(other.max),
        }
    }

    /// Compute the AABB of a transformed AABB (uses the 8 corner approach).
    pub fn from_transformed_aabb(local: &Aabb, transform: &Affine3A) -> Aabb {
        let mat = Mat4::from(*transform);
        let corners = [
            Vec3::new(local.min.x, local.min.y, local.min.z),
            Vec3::new(local.max.x, local.min.y, local.min.z),
            Vec3::new(local.min.x, local.max.y, local.min.z),
            Vec3::new(local.min.x, local.min.y, local.max.z),
            Vec3::new(local.max.x, local.max.y, local.min.z),
            Vec3::new(local.min.x, local.max.y, local.max.z),
            Vec3::new(local.max.x, local.min.y, local.max.z),
            Vec3::new(local.max.x, local.max.y, local.max.z),
        ];
        let mut out = Aabb::empty();
        for c in &corners {
            let t = mat.transform_point3(*c);
            out.expand(t);
        }
        out
    }

    pub fn center(&self) -> Vec3 {
        (self.min + self.max) * 0.5
    }

    pub fn extents(&self) -> Vec3 {
        self.max - self.min
    }

    pub fn diagonal(&self) -> f32 {
        self.extents().length()
    }

    pub fn is_valid(&self) -> bool {
        self.min.x <= self.max.x && self.min.y <= self.max.y && self.min.z <= self.max.z
    }
}

impl Default for Aabb {
    fn default() -> Self {
        Self::empty()
    }
}
