{
	"patcher" : 	{
		"fileversion" : 1,
		"appversion" : { "major" : 8, "minor" : 6, "revision" : 0, "architecture" : "x64", "modernui" : 1 },
		"classnamespace" : "box",
		"rect" : [ 100.0, 100.0, 750.0, 550.0 ],
		"description" : "RGB channel swapping in tile regions",
		"digest" : "Applies channel permutation in masked regions",
		"tags" : "jitter, GPU, channel, swap, glitch, effect",
		"boxes" : [
			{ "box" : { "id" : "obj-1", "maxclass" : "comment", "numinlets" : 1, "numoutlets" : 0, "patching_rect" : [ 30.0, 20.0, 500.0, 20.0 ], "text" : "sr.bandswap - RGB Channel Swapping Effect", "fontsize" : 14.0, "fontface" : 1 } },
			{ "box" : { "id" : "obj-2", "maxclass" : "comment", "numinlets" : 1, "numoutlets" : 0, "patching_rect" : [ 30.0, 45.0, 600.0, 40.0 ], "text" : "Swaps RGB channels in tile regions. Works with sr.tilegen for random tiles.\nperm_r/g/b: 0=R, 1=G, 2=B - controls which input channel goes to output" } },
			{ "box" : { "id" : "obj-3", "maxclass" : "toggle", "numinlets" : 1, "numoutlets" : 1, "outlettype" : [ "int" ], "patching_rect" : [ 30.0, 100.0, 24.0, 24.0 ] } },
			{ "box" : { "id" : "obj-4", "maxclass" : "newobj", "numinlets" : 2, "numoutlets" : 1, "outlettype" : [ "bang" ], "patching_rect" : [ 30.0, 130.0, 65.0, 22.0 ], "text" : "qmetro 30" } },
			{ "box" : { "id" : "obj-5", "maxclass" : "newobj", "numinlets" : 1, "numoutlets" : 2, "outlettype" : [ "jit_gl_texture", "" ], "patching_rect" : [ 30.0, 160.0, 280.0, 22.0 ], "text" : "jit.movie @autostart 1 @loop 1 @output_texture 1" } },
			{ "box" : { "id" : "obj-20", "maxclass" : "newobj", "numinlets" : 1, "numoutlets" : 1, "outlettype" : [ "bang" ], "patching_rect" : [ 250.0, 130.0, 58.0, 22.0 ], "text" : "loadbang" } },
			{ "box" : { "id" : "obj-21", "maxclass" : "message", "numinlets" : 2, "numoutlets" : 1, "outlettype" : [ "" ], "patching_rect" : [ 250.0, 160.0, 120.0, 22.0 ], "text" : "read chicken.mp4" } },
			{ "box" : { "id" : "obj-6", "maxclass" : "comment", "numinlets" : 1, "numoutlets" : 0, "patching_rect" : [ 400.0, 100.0, 300.0, 60.0 ], "text" : "Channel mapping:\nperm_r=2, perm_g=1, perm_b=0 = BGR swap\nperm_r=1, perm_g=0, perm_b=2 = GRB swap" } },
			{ "box" : { "id" : "obj-7", "maxclass" : "newobj", "numinlets" : 2, "numoutlets" : 2, "outlettype" : [ "jit_gl_texture", "" ], "patching_rect" : [ 30.0, 210.0, 350.0, 22.0 ], "text" : "jit.gl.pix @gen sr.bandswap @perm_r 2 @perm_g 1 @perm_b 0" } },
			{ "box" : { "id" : "obj-8", "maxclass" : "jit.pwindow", "numinlets" : 1, "numoutlets" : 2, "outlettype" : [ "jit_matrix", "" ], "patching_rect" : [ 30.0, 280.0, 320.0, 180.0 ] } },
			{ "box" : { "id" : "obj-9", "maxclass" : "comment", "numinlets" : 1, "numoutlets" : 0, "patching_rect" : [ 400.0, 280.0, 300.0, 80.0 ], "text" : "Requires tile mask on second input.\nUse sr.tilegen to generate random tiles,\nthen render to texture for mask input.\n\nWithout mask, entire image is swapped." } }
		],
		"lines" : [
			{ "patchline" : { "source" : [ "obj-20", 0 ], "destination" : [ "obj-21", 0 ] } },
			{ "patchline" : { "source" : [ "obj-21", 0 ], "destination" : [ "obj-5", 0 ] } },
			{ "patchline" : { "source" : [ "obj-3", 0 ], "destination" : [ "obj-4", 0 ] } },
			{ "patchline" : { "source" : [ "obj-4", 0 ], "destination" : [ "obj-5", 0 ] } },
			{ "patchline" : { "source" : [ "obj-5", 0 ], "destination" : [ "obj-7", 0 ] } },
			{ "patchline" : { "source" : [ "obj-7", 0 ], "destination" : [ "obj-8", 0 ] } }
		]
	}
}
