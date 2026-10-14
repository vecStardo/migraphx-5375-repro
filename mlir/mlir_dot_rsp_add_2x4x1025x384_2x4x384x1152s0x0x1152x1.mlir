module {
  func.func @mlir_dot_reshape_add(%arg0: !migraphx.shaped<2x4x1025x384xf32, 1574400x393600x384x1>, %arg1: !migraphx.shaped<384x1152xf32, 1152x1>, %arg2: !migraphx.shaped<1025x1152xf32, 1152x1>) -> !migraphx.shaped<8x1025x1152xf32, 1180800x1152x1> attributes {rock.arch = "", rock.kernel = "mixr", rock.num_chiplets = 0 : i64, rock.num_cu = 0 : i64} {
    %0 = migraphx.multibroadcast %arg1 {out_dyn_dims = [], out_lens = [2, 4, 384, 1152]} : <384x1152xf32, 1152x1> -> <2x4x384x1152xf32, 0x0x1152x1>
    %1 = migraphx.dot %arg0, %0 : <2x4x1025x384xf32, 1574400x393600x384x1>, <2x4x384x1152xf32, 0x0x1152x1> -> <2x4x1025x1152xf32, 4723200x1180800x1152x1>
    %2 = migraphx.reshape %1 {dims = [8, 1025, 1152]} : <2x4x1025x1152xf32, 4723200x1180800x1152x1> -> <8x1025x1152xf32, 1180800x1152x1>
    %3 = migraphx.multibroadcast %arg2 {out_dyn_dims = [], out_lens = [8, 1025, 1152]} : <1025x1152xf32, 1152x1> -> <8x1025x1152xf32, 0x1152x1>
    %4 = migraphx.add %2, %3 : <8x1025x1152xf32, 1180800x1152x1>, <8x1025x1152xf32, 0x1152x1> -> <8x1025x1152xf32, 1180800x1152x1>
    return %4 : !migraphx.shaped<8x1025x1152xf32, 1180800x1152x1>
  }
}