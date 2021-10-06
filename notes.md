# Points
## Vertical
### Nose
- 28
- 29
- 30
- 27
### Eye
+ 38
+ 43
- 40
- 47
### Mouth
- 49
- 53

+ 60
+ 64

- 56
- 57
- 58

+ 55
+ 59
### Jaw
- 0
- 16
### Brow
+ 21
+ 22

## Horizontal
### Eye
- 42
+ 39
+ 43
- 38
### Nose
- 31
+ 35
- 32
+ 34
### Brow
- 18
- 21
+ 22
+ 25
### Mouth
+ 53
- 49
### Jaw
- 5
+ 11

# Features
MLINE = Ave(y(61, 62, 63))
MLINE2 = Ave(y(65, 66, 67))
VLINE = Ave(x(27, 28, 29, 30, 33, 8))
## Vertical
0. Nose height: Ave(y(27, 28, 29, 30))\*
1. Eye height: Ave(y(40) - y(38), y(47) - y(40))
2. Upper lip: Ave(y(49, 53)) - MLINE
3. Jaw: Ave(y(0, 16))\*
4. Brow: Ave(y(21, 22))\*
5. Mouth: Ave(y(60, 64)) - MLINE
6. Lower lip, outer: Ave(y(55, 59)) - MLINE
7. Lower lip, mid: Ave(y(56, 57, 58)) - MLINE2
## Horizontal
8. Eye: x(39)-x(42)
9. Nostrils: Ave(x(31, 32)) - Ave(x(34,35))
10. Eyelid: x(38)-x(43)
11. Inner brow: x(21)-x(22)
12. Outer brow: x(18)-x(25)
13. Upper lip: x(49)-x(53)
14. Lower jaw: x(5)-x(11)
