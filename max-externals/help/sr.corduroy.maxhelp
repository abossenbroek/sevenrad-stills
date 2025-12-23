{
	"patcher" : 	{
		"fileversion" : 1,
		"appversion" : { "major" : 8, "minor" : 6, "revision" : 0, "architecture" : "x64", "modernui" : 1 },
		"classnamespace" : "box",
		"rect" : [ 100.0, 100.0, 750.0, 550.0 ],
		"description" : "Corduroy striping artifact from scanner calibration drift",
		"digest" : "Simulates push-broom scanner banding artifacts",
		"tags" : "jitter, GPU, corduroy, scanner, satellite, effect",
		"boxes" : [
			{ "box" : { "id" : "obj-1", "maxclass" : "comment", "numinlets" : 1, "numoutlets" : 0, "patching_rect" : [ 30.0, 20.0, 400.0, 20.0 ], "text" : "sr.corduroy - Scanner Banding Effect", "fontsize" : 14.0, "fontface" : 1 } },
			{ "box" : { "id" : "obj-2", "maxclass" : "comment", "numinlets" : 1, "numoutlets" : 0, "patching_rect" : [ 30.0, 45.0, 600.0, 40.0 ], "text" : "Simulates push-broom/whisk-broom scanner calibration drift.\nCreates striping artifacts where detector elements have varying sensitivity." } },
			{ "box" : { "id" : "obj-3", "maxclass" : "toggle", "numinlets" : 1, "numoutlets" : 1, "outlettype" : [ "int" ], "patching_rect" : [ 30.0, 100.0, 24.0, 24.0 ] } },
			{ "box" : { "id" : "obj-4", "maxclass" : "newobj", "numinlets" : 2, "numoutlets" : 1, "outlettype" : [ "bang" ], "patching_rect" : [ 30.0, 130.0, 65.0, 22.0 ], "text" : "qmetro 30" } },
			{ "box" : { "id" : "obj-5", "maxclass" : "newobj", "numinlets" : 1, "numoutlets" : 2, "outlettype" : [ "jit_matrix", "" ], "patching_rect" : [ 30.0, 160.0, 200.0, 22.0 ], "text" : "jit.movie @autostart 1 @loop 1" } },
			{ "box" : { "id" : "obj-6", "maxclass" : "comment", "numinlets" : 1, "numoutlets" : 0, "patching_rect" : [ 400.0, 100.0, 200.0, 20.0 ], "text" : "orientation: 0=vertical, 1=horizontal" } },
			{ "box" : { "id" : "obj-7", "maxclass" : "number", "numinlets" : 1, "numoutlets" : 2, "outlettype" : [ "", "bang" ], "patching_rect" : [ 400.0, 125.0, 50.0, 22.0 ], "minimum" : 0, "maximum" : 1 } },
			{ "box" : { "id" : "obj-8", "maxclass" : "message", "numinlets" : 2, "numoutlets" : 1, "outlettype" : [ "" ], "patching_rect" : [ 400.0, 155.0, 100.0, 22.0 ], "text" : "orientation $1" } },
			{ "box" : { "id" : "obj-9", "maxclass" : "comment", "numinlets" : 1, "numoutlets" : 0, "patching_rect" : [ 520.0, 100.0, 120.0, 20.0 ], "text" : "strength: 0.0-1.0" } },
			{ "box" : { "id" : "obj-10", "maxclass" : "flonum", "numinlets" : 1, "numoutlets" : 2, "outlettype" : [ "", "bang" ], "patching_rect" : [ 520.0, 125.0, 60.0, 22.0 ], "minimum" : 0.0, "maximum" : 1.0 } },
			{ "box" : { "id" : "obj-11", "maxclass" : "message", "numinlets" : 2, "numoutlets" : 1, "outlettype" : [ "" ], "patching_rect" : [ 520.0, 155.0, 80.0, 22.0 ], "text" : "strength $1" } },
			{ "box" : { "id" : "obj-12", "maxclass" : "comment", "numinlets" : 1, "numoutlets" : 0, "patching_rect" : [ 620.0, 100.0, 100.0, 20.0 ], "text" : "density: 0.0-1.0" } },
			{ "box" : { "id" : "obj-13", "maxclass" : "flonum", "numinlets" : 1, "numoutlets" : 2, "outlettype" : [ "", "bang" ], "patching_rect" : [ 620.0, 125.0, 60.0, 22.0 ], "minimum" : 0.0, "maximum" : 1.0 } },
			{ "box" : { "id" : "obj-14", "maxclass" : "message", "numinlets" : 2, "numoutlets" : 1, "outlettype" : [ "" ], "patching_rect" : [ 620.0, 155.0, 80.0, 22.0 ], "text" : "density $1" } },
			{ "box" : { "id" : "obj-15", "maxclass" : "newobj", "numinlets" : 1, "numoutlets" : 2, "outlettype" : [ "jit_gl_texture", "" ], "patching_rect" : [ 30.0, 210.0, 350.0, 22.0 ], "text" : "jit.gl.pix @gen sr.corduroy @orientation 0 @strength 0.3 @density 0.2" } },
			{ "box" : { "id" : "obj-16", "maxclass" : "jit.pwindow", "numinlets" : 1, "numoutlets" : 2, "outlettype" : [ "jit_matrix", "" ], "patching_rect" : [ 30.0, 260.0, 320.0, 180.0 ] } }
		],
		"lines" : [
			{ "patchline" : { "source" : [ "obj-3", 0 ], "destination" : [ "obj-4", 0 ] } },
			{ "patchline" : { "source" : [ "obj-4", 0 ], "destination" : [ "obj-5", 0 ] } },
			{ "patchline" : { "source" : [ "obj-5", 0 ], "destination" : [ "obj-15", 0 ] } },
			{ "patchline" : { "source" : [ "obj-7", 0 ], "destination" : [ "obj-8", 0 ] } },
			{ "patchline" : { "source" : [ "obj-8", 0 ], "destination" : [ "obj-15", 0 ] } },
			{ "patchline" : { "source" : [ "obj-10", 0 ], "destination" : [ "obj-11", 0 ] } },
			{ "patchline" : { "source" : [ "obj-11", 0 ], "destination" : [ "obj-15", 0 ] } },
			{ "patchline" : { "source" : [ "obj-13", 0 ], "destination" : [ "obj-14", 0 ] } },
			{ "patchline" : { "source" : [ "obj-14", 0 ], "destination" : [ "obj-15", 0 ] } },
			{ "patchline" : { "source" : [ "obj-15", 0 ], "destination" : [ "obj-16", 0 ] } }
		]
	}
}
