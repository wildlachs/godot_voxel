let SessionLoad = 1
let s:so_save = &g:so | let s:siso_save = &g:siso | setg so=0 siso=0 | setl so=-1 siso=-1
let v:this_session=expand("<sfile>:p")
silent only
silent tabonly
cd ~/Projekte/VoxelLighting/godot/modules/voxel
if expand('%') == '' && !&modified && line('$') <= 1 && getline(1) == ''
  let s:wipebuf = bufnr('%')
endif
let s:shortmess_save = &shortmess
if &shortmess =~ 'A'
  set shortmess=aoOA
else
  set shortmess=aoO
endif
badd +2827 terrain/fixed_lod/voxel_terrain.cpp
badd +193 meshers/blocky/voxel_blocky_model.h
badd +252 meshers/blocky/voxel_blocky_model.cpp
badd +175 meshers/blocky/blocky_baked_library.h
badd +100 doc/classes/VoxelBlockyModel.xml
badd +128 meshers/blocky/blocky_lod_skirts.h
badd +426 meshers/blocky/types/voxel_blocky_type.cpp
badd +292 meshers/blocky/voxel_blocky_model_cube.cpp
badd +522 meshers/blocky/voxel_blocky_model_mesh.cpp
badd +17 meshers/blocky/voxel_mesher_blocky.cpp
badd +56 util/rgblight.h
badd +44 meshers/mesh_block_task.h
badd +321 meshers/mesh_block_task.cpp
badd +51 terrain/fixed_lod/voxel_terrain.h
badd +73 ~/Projekte/VoxelLighting/godot/core/math/vector3i.h
badd +51 meshers/voxel_mesher.h
badd +712 streams/region/region_file.cpp
badd +28 meshers/voxel_mesher.cpp
badd +1193 meshers/cubes/voxel_mesher_cubes.cpp
argglobal
%argdel
edit meshers/mesh_block_task.cpp
let s:save_splitbelow = &splitbelow
let s:save_splitright = &splitright
set splitbelow splitright
wincmd _ | wincmd |
vsplit
1wincmd h
wincmd w
let &splitbelow = s:save_splitbelow
let &splitright = s:save_splitright
wincmd t
let s:save_winminheight = &winminheight
let s:save_winminwidth = &winminwidth
set winminheight=0
set winheight=1
set winminwidth=0
set winwidth=1
exe 'vert 1resize ' . ((&columns * 106 + 106) / 213)
exe 'vert 2resize ' . ((&columns * 106 + 106) / 213)
argglobal
balt meshers/mesh_block_task.h
setlocal fdm=manual
setlocal fde=0
setlocal fmr={{{,}}}
setlocal fdi=#
setlocal fdl=0
setlocal fml=1
setlocal fdn=20
setlocal fen
silent! normal! zE
let &fdl = &fdl
let s:l = 321 - ((11 * winheight(0) + 26) / 52)
if s:l < 1 | let s:l = 1 | endif
keepjumps exe s:l
normal! zt
keepjumps 321
normal! 021|
wincmd w
argglobal
if bufexists(fnamemodify("terrain/fixed_lod/voxel_terrain.cpp", ":p")) | buffer terrain/fixed_lod/voxel_terrain.cpp | else | edit terrain/fixed_lod/voxel_terrain.cpp | endif
if &buftype ==# 'terminal'
  silent file terrain/fixed_lod/voxel_terrain.cpp
endif
balt meshers/voxel_mesher.cpp
setlocal fdm=manual
setlocal fde=0
setlocal fmr={{{,}}}
setlocal fdi=#
setlocal fdl=0
setlocal fml=1
setlocal fdn=20
setlocal fen
silent! normal! zE
let &fdl = &fdl
let s:l = 2827 - ((6 * winheight(0) + 26) / 52)
if s:l < 1 | let s:l = 1 | endif
keepjumps exe s:l
normal! zt
keepjumps 2827
normal! 09|
wincmd w
exe 'vert 1resize ' . ((&columns * 106 + 106) / 213)
exe 'vert 2resize ' . ((&columns * 106 + 106) / 213)
tabnext 1
if exists('s:wipebuf') && len(win_findbuf(s:wipebuf)) == 0 && getbufvar(s:wipebuf, '&buftype') isnot# 'terminal'
  silent exe 'bwipe ' . s:wipebuf
endif
unlet! s:wipebuf
set winheight=1 winwidth=20
let &shortmess = s:shortmess_save
let &winminheight = s:save_winminheight
let &winminwidth = s:save_winminwidth
let s:sx = expand("<sfile>:p:r")."x.vim"
if filereadable(s:sx)
  exe "source " . fnameescape(s:sx)
endif
let &g:so = s:so_save | let &g:siso = s:siso_save
set hlsearch
nohlsearch
doautoall SessionLoadPost
unlet SessionLoad
" vim: set ft=vim :
