module {
  func.func @mlir_squeeze_dot_transpose_add(%arg0: !migraphx.shaped<1x2x200x256xf32, 102400x51200x256x1>, %arg1: !migraphx.shaped<256x256xf32, 256x1>, %arg2: !migraphx.shaped<256x200xf32, 1x256>) -> !migraphx.shaped<2x256x200xf32, 51200x1x256> attributes {rock.arch = "", rock.kernel = "mixr", rock.num_chiplets = 0 : i64, rock.num_cu = 0 : i64} {
    %0 = migraphx.reshape %arg0 {dims = [2, 200, 256]} : <1x2x200x256xf32, 102400x51200x256x1> -> <2x200x256xf32, 51200x256x1>
    %1 = migraphx.multibroadcast %arg1 {out_dyn_dims = [], out_lens = [2, 256, 256]} : <256x256xf32, 256x1> -> <2x256x256xf32, 0x256x1>
    %2 = migraphx.dot %0, %1 : <2x200x256xf32, 51200x256x1>, <2x256x256xf32, 0x256x1> -> <2x200x256xf32, 51200x256x1>
    %3 = migraphx.transpose %2 {permutation = [0, 2, 1]} : <2x200x256xf32, 51200x256x1> -> <2x256x200xf32, 51200x1x256>
    %4 = migraphx.multibroadcast %arg2 {out_dyn_dims = [], out_lens = [2, 256, 200]} : <256x200xf32, 1x256> -> <2x256x200xf32, 0x1x256>
    %5 = migraphx.add %3, %4 : <2x256x200xf32, 51200x1x256>, <2x256x200xf32, 0x1x256> -> <2x256x200xf32, 51200x1x256>
    return %5 : !migraphx.shaped<2x256x200xf32, 51200x1x256>
  }
}