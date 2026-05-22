// Expects fixed ISO 8601 format: "YYYY-MM-DDTHH:MM:SSZ" (20 bytes)

pub fn get_hour(date: &str) -> f64 {
    parse_u8(date.as_bytes(), 11) as f64 / 23.0
}

pub fn get_day_of_week(date: &str) -> f64 {
    let b = date.as_bytes();
    let y = parse_u16(b, 0) as i32;
    let m = parse_u8(b, 5) as i32;
    let d = parse_u8(b, 8) as i32;

    // Tomohiko Sakamoto's algorithm — returns 0=Sun..6=Sat
    let t: [i32; 12] = [0, 3, 2, 5, 0, 3, 5, 1, 4, 6, 2, 4];
    let y = if m < 3 { y - 1 } else { y };
    let dow = (y + y / 4 - y / 100 + y / 400 + t[(m - 1) as usize] + d) % 7;
    let dow_mon = (dow + 6) % 7; // remap: Mon=0..Sun=6
    dow_mon as f64 / 6.0
}

pub fn minutes_between(from: &str, to: &str) -> f64 {
    (to_total_minutes(to.as_bytes()) - to_total_minutes(from.as_bytes())).max(0) as f64
}

fn to_total_minutes(b: &[u8]) -> i64 {
    let y = parse_u16(b, 0) as i64;
    let m = parse_u8(b, 5) as i64;
    let d = parse_u8(b, 8) as i64;
    let h = parse_u8(b, 11) as i64;
    let min = parse_u8(b, 14) as i64;

    let month_days: [i64; 12] = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334];
    let is_leap = (y % 4 == 0 && y % 100 != 0) || (y % 400 == 0);
    let leap_offset = if is_leap && m > 2 { 1 } else { 0 };
    let idx = ((m - 1).clamp(0, 11)) as usize;

    let total_days = y * 365 + (y - 1) / 4 - (y - 1) / 100 + (y - 1) / 400
        + month_days[idx]
        + leap_offset
        + (d - 1);

    total_days * 1440 + h * 60 + min
}

#[inline(always)]
fn parse_u8(b: &[u8], offset: usize) -> u8 {
    (b[offset] - b'0') * 10 + (b[offset + 1] - b'0')
}

#[inline(always)]
fn parse_u16(b: &[u8], offset: usize) -> u16 {
    (b[offset] - b'0') as u16 * 1000
        + (b[offset + 1] - b'0') as u16 * 100
        + (b[offset + 2] - b'0') as u16 * 10
        + (b[offset + 3] - b'0') as u16
}
