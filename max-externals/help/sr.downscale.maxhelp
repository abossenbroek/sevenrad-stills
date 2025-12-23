{
	"patcher" : 	{
		"fileversion" : 1,
		"appversion" : { "major" : 8, "minor" : 6, "revision" : 0, "architecture" : "x64", "modernui" : 1 },
		"classnamespace" : "box",
		"rect" : [ 100.0, 100.0, 700.0, 500.0 ],
		"description" : "Pixelation/downscale effect",
		"digest" : "Creates retro pixelated look by reducing effective resolution",
		"tags" : "jitter, GPU, pixelate, downscale, retro, effect",
		"boxes" : [
			{ "box" : { "id" : "obj-1", "maxclass" : "comment", "numinlets" : 1, "numoutlets" : 0, "patching_rect" : [ 30.0, 20.0, 400.0, 20.0 ], "text" : "sr.downscale - Pixelation Effect", "fontsize" : 14.0, "fontface" : 1 } },
			{ "box" : { "id" : "obj-2", "maxclass" : "comment", "numinlets" : 1, "numoutlets" : 0, "patching_rect" : [ 30.0, 45.0, 500.0, 40.0 ], "text" : "Reduces effective resolution to create pixelated/retro look.\nscale 0.5 = half resolution, scale 0.1 = heavy pixelation" } },
			{ "box" : { "id" : "obj-3", "maxclass" : "toggle", "numinlets" : 1, "numoutlets" : 1, "outlettype" : [ "int" ], "patching_rect" : [ 30.0, 100.0, 24.0, 24.0 ] } },
			{ "box" : { "id" : "obj-4", "maxclass" : "newobj", "numinlets" : 2, "numoutlets" : 1, "outlettype" : [ "bang" ], "patching_rect" : [ 30.0, 130.0, 65.0, 22.0 ], "text" : "qmetro 30" } },
			{ "box" : { "id" : "obj-5", "maxclass" : "newobj", "numinlets" : 1, "numoutlets" : 2, "outlettype" : [ "jit_matrix", "" ], "patching_rect" : [ 30.0, 160.0, 200.0, 22.0 ], "text" : "jit.movie @autostart 1 @loop 1" } },
			{ "box" : { "id" : "obj-6", "maxclass" : "comment", "numinlets" : 1, "numoutlets" : 0, "patching_rect" : [ 400.0, 100.0, 150.0, 20.0 ], "text" : "scale: 0.01 - 1.0" } },
			{ "box" : { "id" : "obj-7", "maxclass" : "flonum", "numinlets" : 1, "numoutlets" : 2, "outlettype" : [ "", "bang" ], "patching_rect" : [ 400.0, 125.0, 60.0, 22.0 ], "minimum" : 0.01, "maximum" : 1.0 } },
			{ "box" : { "id" : "obj-8", "maxclass" : "message", "numinlets" : 2, "numoutlets" : 1, "outlettype" : [ "" ], "patching_rect" : [ 400.0, 155.0, 80.0, 22.0 ], "text" : "scale $1" } },
			{ "box" : { "id" : "obj-9", "maxclass" : "newobj", "numinlets" : 1, "numoutlets" : 2, "outlettype" : [ "jit_gl_texture", "" ], "patching_rect" : [ 30.0, 210.0, 280.0, 22.0 ], "text" : "jit.gl.pix @gen sr.downscale @scale 0.25" } },
			{ "box" : { "id" : "obj-10", "maxclass" : "jit.pwindow", "numinlets" : 1, "numoutlets" : 2, "outlettype" : [ "jit_matrix", "" ], "patching_rect" : [ 30.0, 260.0, 320.0, 180.0 ] } }
		],
		"lines" : [
			{ "patchline" : { "source" : [ "obj-3", 0 ], "destination" : [ "obj-4", 0 ] } },
			{ "patchline" : { "source" : [ "obj-4", 0 ], "destination" : [ "obj-5", 0 ] } },
			{ "patchline" : { "source" : [ "obj-5", 0 ], "destination" : [ "obj-9", 0 ] } },
			{ "patchline" : { "source" : [ "obj-7", 0 ], "destination" : [ "obj-8", 0 ] } },
			{ "patchline" : { "source" : [ "obj-8", 0 ], "destination" : [ "obj-9", 0 ] } },
			{ "patchline" : { "source" : [ "obj-9", 0 ], "destination" : [ "obj-10", 0 ] } }
		]
	}
}
