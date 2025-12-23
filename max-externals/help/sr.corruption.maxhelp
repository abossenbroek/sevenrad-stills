{
	"patcher" : 	{
		"fileversion" : 1,
		"appversion" : { "major" : 8, "minor" : 6, "revision" : 0, "architecture" : "x64", "modernui" : 1 },
		"classnamespace" : "box",
		"rect" : [ 100.0, 100.0, 800.0, 600.0 ],
		"description" : "Buffer corruption effects simulating cosmic ray memory upsets",
		"digest" : "XOR, invert, and shuffle corruption in tile regions",
		"tags" : "jitter, GPU, corruption, glitch, satellite, effect",
		"boxes" : [
			{ "box" : { "id" : "obj-1", "maxclass" : "comment", "numinlets" : 1, "numoutlets" : 0, "patching_rect" : [ 30.0, 20.0, 500.0, 20.0 ], "text" : "sr.corruption - Buffer Corruption Effects", "fontsize" : 14.0, "fontface" : 1 } },
			{ "box" : { "id" : "obj-2", "maxclass" : "comment", "numinlets" : 1, "numoutlets" : 0, "patching_rect" : [ 30.0, 45.0, 700.0, 40.0 ], "text" : "Simulates cosmic ray hits causing single-event upsets in satellite memory.\nWorks with tile mask from sr.tilegen. Three corruption modes available." } },
			{ "box" : { "id" : "obj-3", "maxclass" : "toggle", "numinlets" : 1, "numoutlets" : 1, "outlettype" : [ "int" ], "patching_rect" : [ 30.0, 100.0, 24.0, 24.0 ] } },
			{ "box" : { "id" : "obj-4", "maxclass" : "newobj", "numinlets" : 2, "numoutlets" : 1, "outlettype" : [ "bang" ], "patching_rect" : [ 30.0, 130.0, 65.0, 22.0 ], "text" : "qmetro 30" } },
			{ "box" : { "id" : "obj-5", "maxclass" : "newobj", "numinlets" : 1, "numoutlets" : 2, "outlettype" : [ "jit_gl_texture", "" ], "patching_rect" : [ 30.0, 160.0, 280.0, 22.0 ], "text" : "jit.movie @autostart 1 @loop 1 @output_texture 1" } },
			{ "box" : { "id" : "obj-20", "maxclass" : "newobj", "numinlets" : 1, "numoutlets" : 1, "outlettype" : [ "bang" ], "patching_rect" : [ 250.0, 130.0, 58.0, 22.0 ], "text" : "loadbang" } },
			{ "box" : { "id" : "obj-21", "maxclass" : "message", "numinlets" : 2, "numoutlets" : 1, "outlettype" : [ "" ], "patching_rect" : [ 250.0, 160.0, 120.0, 22.0 ], "text" : "read chicken.mp4" } },
			{ "box" : { "id" : "obj-6", "maxclass" : "comment", "numinlets" : 1, "numoutlets" : 0, "patching_rect" : [ 450.0, 100.0, 200.0, 40.0 ], "text" : "mode:\n0=XOR, 1=invert, 2=shuffle" } },
			{ "box" : { "id" : "obj-7", "maxclass" : "number", "numinlets" : 1, "numoutlets" : 2, "outlettype" : [ "", "bang" ], "patching_rect" : [ 450.0, 150.0, 50.0, 22.0 ], "minimum" : 0, "maximum" : 2 } },
			{ "box" : { "id" : "obj-8", "maxclass" : "message", "numinlets" : 2, "numoutlets" : 1, "outlettype" : [ "" ], "patching_rect" : [ 450.0, 180.0, 80.0, 22.0 ], "text" : "mode $1" } },
			{ "box" : { "id" : "obj-9", "maxclass" : "comment", "numinlets" : 1, "numoutlets" : 0, "patching_rect" : [ 550.0, 100.0, 100.0, 20.0 ], "text" : "intensity: 0.0-1.0" } },
			{ "box" : { "id" : "obj-10", "maxclass" : "flonum", "numinlets" : 1, "numoutlets" : 2, "outlettype" : [ "", "bang" ], "patching_rect" : [ 550.0, 125.0, 60.0, 22.0 ], "minimum" : 0.0, "maximum" : 1.0 } },
			{ "box" : { "id" : "obj-11", "maxclass" : "message", "numinlets" : 2, "numoutlets" : 1, "outlettype" : [ "" ], "patching_rect" : [ 550.0, 155.0, 80.0, 22.0 ], "text" : "intensity $1" } },
			{ "box" : { "id" : "obj-12", "maxclass" : "comment", "numinlets" : 1, "numoutlets" : 0, "patching_rect" : [ 660.0, 100.0, 80.0, 20.0 ], "text" : "seed: integer" } },
			{ "box" : { "id" : "obj-13", "maxclass" : "number", "numinlets" : 1, "numoutlets" : 2, "outlettype" : [ "", "bang" ], "patching_rect" : [ 660.0, 125.0, 60.0, 22.0 ] } },
			{ "box" : { "id" : "obj-14", "maxclass" : "message", "numinlets" : 2, "numoutlets" : 1, "outlettype" : [ "" ], "patching_rect" : [ 660.0, 155.0, 70.0, 22.0 ], "text" : "seed $1" } },
			{ "box" : { "id" : "obj-15", "maxclass" : "newobj", "numinlets" : 2, "numoutlets" : 2, "outlettype" : [ "jit_gl_texture", "" ], "patching_rect" : [ 30.0, 250.0, 350.0, 22.0 ], "text" : "jit.gl.pix @gen sr.corruption @mode 0 @intensity 0.5 @seed 42" } },
			{ "box" : { "id" : "obj-16", "maxclass" : "jit.pwindow", "numinlets" : 1, "numoutlets" : 2, "outlettype" : [ "jit_matrix", "" ], "patching_rect" : [ 30.0, 320.0, 320.0, 180.0 ] } },
			{ "box" : { "id" : "obj-17", "maxclass" : "comment", "numinlets" : 1, "numoutlets" : 0, "patching_rect" : [ 400.0, 320.0, 350.0, 120.0 ], "text" : "Corruption Modes:\n\nXOR (0): Simulates bitwise XOR with random noise\nInvert (1): Blends with inverted colors\nShuffle (2): Random per-pixel channel permutation\n\nRequires tile mask on second input from sr.tilegen." } }
		],
		"lines" : [
			{ "patchline" : { "source" : [ "obj-20", 0 ], "destination" : [ "obj-21", 0 ] } },
			{ "patchline" : { "source" : [ "obj-21", 0 ], "destination" : [ "obj-5", 0 ] } },
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
