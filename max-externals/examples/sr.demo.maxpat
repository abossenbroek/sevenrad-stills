{
	"patcher" : 	{
		"fileversion" : 1,
		"appversion" : 		{
			"major" : 8,
			"minor" : 6,
			"revision" : 0,
			"architecture" : "x64",
			"modernui" : 1
		},
		"classnamespace" : "box",
		"rect" : [ 50.0, 50.0, 1200.0, 800.0 ],
		"bglocked" : 0,
		"openinpresentation" : 0,
		"default_fontsize" : 12.0,
		"default_fontface" : 0,
		"default_fontname" : "Arial",
		"gridonopen" : 1,
		"gridsize" : [ 15.0, 15.0 ],
		"gridsnaponopen" : 1,
		"objectsnaponopen" : 1,
		"statusbarvisible" : 2,
		"toolbarvisible" : 1,
		"lefttoolbarpinned" : 0,
		"toptoolbarpinned" : 0,
		"righttoolbarpinned" : 0,
		"bottomtoolbarpinned" : 0,
		"toolbars_unpinned_last_save" : 0,
		"tallnewobj" : 0,
		"boxanimatetime" : 200,
		"enablehscroll" : 1,
		"enablevscroll" : 1,
		"devicewidth" : 0.0,
		"description" : "SevenRad Effects Chain Demonstration",
		"digest" : "Demonstrates chaining multiple SevenRad GPU effects",
		"tags" : "jitter, GPU, sevenrad, effects, demo",
		"style" : "",
		"subpatcher_template" : "",
		"assistshowspatchername" : 0,
		"boxes" : [ 			{
				"box" : 				{
					"id" : "obj-1",
					"maxclass" : "comment",
					"numinlets" : 1,
					"numoutlets" : 0,
					"patching_rect" : [ 30.0, 20.0, 600.0, 20.0 ],
					"text" : "SevenRad Effects Chain Demo - Noise → Saturation → Chromatic → Corduroy",
					"fontsize" : 14.0,
					"fontface" : 1
				}
			},
			{
				"box" : 				{
					"id" : "obj-2",
					"maxclass" : "comment",
					"numinlets" : 1,
					"numoutlets" : 0,
					"patching_rect" : [ 30.0, 45.0, 700.0, 60.0 ],
					"text" : "This patcher demonstrates chaining multiple SevenRad effects together.\nSignal Flow: Video Input → sr.noise → sr.saturation → sr.chromatic → sr.corduroy → Output\nEach effect processes the output of the previous effect, creating complex glitch aesthetics."
				}
			},
			{
				"box" : 				{
					"id" : "obj-3",
					"maxclass" : "comment",
					"numinlets" : 1,
					"numoutlets" : 0,
					"patching_rect" : [ 30.0, 110.0, 200.0, 20.0 ],
					"text" : "VIDEO INPUT",
					"fontsize" : 12.0,
					"fontface" : 1
				}
			},
			{
				"box" : 				{
					"id" : "obj-4",
					"maxclass" : "toggle",
					"numinlets" : 1,
					"numoutlets" : 1,
					"outlettype" : [ "int" ],
					"patching_rect" : [ 30.0, 135.0, 24.0, 24.0 ]
				}
			},
			{
				"box" : 				{
					"id" : "obj-5",
					"maxclass" : "newobj",
					"numinlets" : 2,
					"numoutlets" : 1,
					"outlettype" : [ "bang" ],
					"patching_rect" : [ 30.0, 165.0, 65.0, 22.0 ],
					"text" : "qmetro 30"
				}
			},
			{
				"box" : 				{
					"id" : "obj-6",
					"maxclass" : "newobj",
					"numinlets" : 1,
					"numoutlets" : 2,
					"outlettype" : [ "jit_matrix", "" ],
					"patching_rect" : [ 30.0, 195.0, 200.0, 22.0 ],
					"text" : "jit.movie @autostart 1 @loop 1"
				}
			},
			{
				"box" : 				{
					"id" : "obj-7",
					"maxclass" : "message",
					"numinlets" : 2,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 250.0, 195.0, 35.0, 22.0 ],
					"text" : "read"
				}
			},
			{
				"box" : 				{
					"id" : "obj-8",
					"maxclass" : "comment",
					"numinlets" : 1,
					"numoutlets" : 0,
					"patching_rect" : [ 30.0, 230.0, 200.0, 20.0 ],
					"text" : "EFFECT 1: NOISE",
					"fontsize" : 12.0,
					"fontface" : 1
				}
			},
			{
				"box" : 				{
					"id" : "obj-9",
					"maxclass" : "comment",
					"numinlets" : 1,
					"numoutlets" : 0,
					"patching_rect" : [ 350.0, 230.0, 200.0, 20.0 ],
					"text" : "Noise Parameters:",
					"fontface" : 1
				}
			},
			{
				"box" : 				{
					"id" : "obj-10",
					"maxclass" : "comment",
					"numinlets" : 1,
					"numoutlets" : 0,
					"patching_rect" : [ 350.0, 255.0, 150.0, 20.0 ],
					"text" : "mode (0=gauss,1=row,2=col):"
				}
			},
			{
				"box" : 				{
					"id" : "obj-11",
					"maxclass" : "number",
					"numinlets" : 1,
					"numoutlets" : 2,
					"outlettype" : [ "", "bang" ],
					"patching_rect" : [ 510.0, 255.0, 50.0, 22.0 ],
					"minimum" : 0,
					"maximum" : 2
				}
			},
			{
				"box" : 				{
					"id" : "obj-12",
					"maxclass" : "message",
					"numinlets" : 2,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 510.0, 285.0, 80.0, 22.0 ],
					"text" : "mode $1"
				}
			},
			{
				"box" : 				{
					"id" : "obj-13",
					"maxclass" : "comment",
					"numinlets" : 1,
					"numoutlets" : 0,
					"patching_rect" : [ 600.0, 255.0, 100.0, 20.0 ],
					"text" : "amount (0.0-1.0):"
				}
			},
			{
				"box" : 				{
					"id" : "obj-14",
					"maxclass" : "flonum",
					"numinlets" : 1,
					"numoutlets" : 2,
					"outlettype" : [ "", "bang" ],
					"patching_rect" : [ 710.0, 255.0, 60.0, 22.0 ],
					"minimum" : 0.0,
					"maximum" : 1.0
				}
			},
			{
				"box" : 				{
					"id" : "obj-15",
					"maxclass" : "message",
					"numinlets" : 2,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 710.0, 285.0, 80.0, 22.0 ],
					"text" : "amount $1"
				}
			},
			{
				"box" : 				{
					"id" : "obj-16",
					"maxclass" : "newobj",
					"numinlets" : 1,
					"numoutlets" : 2,
					"outlettype" : [ "jit_gl_texture", "" ],
					"patching_rect" : [ 30.0, 315.0, 330.0, 22.0 ],
					"text" : "jit.gl.pix @gen sr.noise @mode 0 @amount 0.15 @seed 42"
				}
			},
			{
				"box" : 				{
					"id" : "obj-17",
					"maxclass" : "comment",
					"numinlets" : 1,
					"numoutlets" : 0,
					"patching_rect" : [ 30.0, 350.0, 200.0, 20.0 ],
					"text" : "EFFECT 2: SATURATION",
					"fontsize" : 12.0,
					"fontface" : 1
				}
			},
			{
				"box" : 				{
					"id" : "obj-18",
					"maxclass" : "comment",
					"numinlets" : 1,
					"numoutlets" : 0,
					"patching_rect" : [ 350.0, 350.0, 200.0, 20.0 ],
					"text" : "Saturation Parameters:",
					"fontface" : 1
				}
			},
			{
				"box" : 				{
					"id" : "obj-19",
					"maxclass" : "comment",
					"numinlets" : 1,
					"numoutlets" : 0,
					"patching_rect" : [ 350.0, 375.0, 150.0, 20.0 ],
					"text" : "factor (0.0-3.0):"
				}
			},
			{
				"box" : 				{
					"id" : "obj-20",
					"maxclass" : "flonum",
					"numinlets" : 1,
					"numoutlets" : 2,
					"outlettype" : [ "", "bang" ],
					"patching_rect" : [ 460.0, 375.0, 60.0, 22.0 ],
					"minimum" : 0.0,
					"maximum" : 3.0
				}
			},
			{
				"box" : 				{
					"id" : "obj-21",
					"maxclass" : "message",
					"numinlets" : 2,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 460.0, 405.0, 80.0, 22.0 ],
					"text" : "factor $1"
				}
			},
			{
				"box" : 				{
					"id" : "obj-22",
					"maxclass" : "newobj",
					"numinlets" : 1,
					"numoutlets" : 2,
					"outlettype" : [ "jit_gl_texture", "" ],
					"patching_rect" : [ 30.0, 435.0, 270.0, 22.0 ],
					"text" : "jit.gl.pix @gen sr.saturation @factor 1.8"
				}
			},
			{
				"box" : 				{
					"id" : "obj-23",
					"maxclass" : "comment",
					"numinlets" : 1,
					"numoutlets" : 0,
					"patching_rect" : [ 30.0, 470.0, 250.0, 20.0 ],
					"text" : "EFFECT 3: CHROMATIC ABERRATION",
					"fontsize" : 12.0,
					"fontface" : 1
				}
			},
			{
				"box" : 				{
					"id" : "obj-24",
					"maxclass" : "comment",
					"numinlets" : 1,
					"numoutlets" : 0,
					"patching_rect" : [ 350.0, 470.0, 200.0, 20.0 ],
					"text" : "Chromatic Parameters:",
					"fontface" : 1
				}
			},
			{
				"box" : 				{
					"id" : "obj-25",
					"maxclass" : "comment",
					"numinlets" : 1,
					"numoutlets" : 0,
					"patching_rect" : [ 350.0, 495.0, 150.0, 20.0 ],
					"text" : "shift_x (-20.0 to 20.0):"
				}
			},
			{
				"box" : 				{
					"id" : "obj-26",
					"maxclass" : "flonum",
					"numinlets" : 1,
					"numoutlets" : 2,
					"outlettype" : [ "", "bang" ],
					"patching_rect" : [ 500.0, 495.0, 60.0, 22.0 ],
					"minimum" : -20.0,
					"maximum" : 20.0
				}
			},
			{
				"box" : 				{
					"id" : "obj-27",
					"maxclass" : "message",
					"numinlets" : 2,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 500.0, 525.0, 80.0, 22.0 ],
					"text" : "shift_x $1"
				}
			},
			{
				"box" : 				{
					"id" : "obj-28",
					"maxclass" : "comment",
					"numinlets" : 1,
					"numoutlets" : 0,
					"patching_rect" : [ 590.0, 495.0, 150.0, 20.0 ],
					"text" : "shift_y (-20.0 to 20.0):"
				}
			},
			{
				"box" : 				{
					"id" : "obj-29",
					"maxclass" : "flonum",
					"numinlets" : 1,
					"numoutlets" : 2,
					"outlettype" : [ "", "bang" ],
					"patching_rect" : [ 740.0, 495.0, 60.0, 22.0 ],
					"minimum" : -20.0,
					"maximum" : 20.0
				}
			},
			{
				"box" : 				{
					"id" : "obj-30",
					"maxclass" : "message",
					"numinlets" : 2,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 740.0, 525.0, 80.0, 22.0 ],
					"text" : "shift_y $1"
				}
			},
			{
				"box" : 				{
					"id" : "obj-31",
					"maxclass" : "newobj",
					"numinlets" : 1,
					"numoutlets" : 2,
					"outlettype" : [ "jit_gl_texture", "" ],
					"patching_rect" : [ 30.0, 555.0, 320.0, 22.0 ],
					"text" : "jit.gl.pix @gen sr.chromatic @shift_x 6.0 @shift_y 0.0"
				}
			},
			{
				"box" : 				{
					"id" : "obj-32",
					"maxclass" : "comment",
					"numinlets" : 1,
					"numoutlets" : 0,
					"patching_rect" : [ 30.0, 590.0, 200.0, 20.0 ],
					"text" : "EFFECT 4: CORDUROY",
					"fontsize" : 12.0,
					"fontface" : 1
				}
			},
			{
				"box" : 				{
					"id" : "obj-33",
					"maxclass" : "comment",
					"numinlets" : 1,
					"numoutlets" : 0,
					"patching_rect" : [ 350.0, 590.0, 200.0, 20.0 ],
					"text" : "Corduroy Parameters:",
					"fontface" : 1
				}
			},
			{
				"box" : 				{
					"id" : "obj-34",
					"maxclass" : "comment",
					"numinlets" : 1,
					"numoutlets" : 0,
					"patching_rect" : [ 350.0, 615.0, 150.0, 20.0 ],
					"text" : "orientation (0=vert,1=horiz):"
				}
			},
			{
				"box" : 				{
					"id" : "obj-35",
					"maxclass" : "number",
					"numinlets" : 1,
					"numoutlets" : 2,
					"outlettype" : [ "", "bang" ],
					"patching_rect" : [ 510.0, 615.0, 50.0, 22.0 ],
					"minimum" : 0,
					"maximum" : 1
				}
			},
			{
				"box" : 				{
					"id" : "obj-36",
					"maxclass" : "message",
					"numinlets" : 2,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 510.0, 645.0, 100.0, 22.0 ],
					"text" : "orientation $1"
				}
			},
			{
				"box" : 				{
					"id" : "obj-37",
					"maxclass" : "comment",
					"numinlets" : 1,
					"numoutlets" : 0,
					"patching_rect" : [ 620.0, 615.0, 150.0, 20.0 ],
					"text" : "strength (0.0-1.0):"
				}
			},
			{
				"box" : 				{
					"id" : "obj-38",
					"maxclass" : "flonum",
					"numinlets" : 1,
					"numoutlets" : 2,
					"outlettype" : [ "", "bang" ],
					"patching_rect" : [ 730.0, 615.0, 60.0, 22.0 ],
					"minimum" : 0.0,
					"maximum" : 1.0
				}
			},
			{
				"box" : 				{
					"id" : "obj-39",
					"maxclass" : "message",
					"numinlets" : 2,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 730.0, 645.0, 80.0, 22.0 ],
					"text" : "strength $1"
				}
			},
			{
				"box" : 				{
					"id" : "obj-40",
					"maxclass" : "newobj",
					"numinlets" : 1,
					"numoutlets" : 2,
					"outlettype" : [ "jit_gl_texture", "" ],
					"patching_rect" : [ 30.0, 675.0, 380.0, 22.0 ],
					"text" : "jit.gl.pix @gen sr.corduroy @orientation 0 @strength 0.35 @density 0.25"
				}
			},
			{
				"box" : 				{
					"id" : "obj-41",
					"maxclass" : "comment",
					"numinlets" : 1,
					"numoutlets" : 0,
					"patching_rect" : [ 30.0, 710.0, 200.0, 20.0 ],
					"text" : "OUTPUT",
					"fontsize" : 12.0,
					"fontface" : 1
				}
			},
			{
				"box" : 				{
					"id" : "obj-42",
					"maxclass" : "jit.pwindow",
					"numinlets" : 1,
					"numoutlets" : 2,
					"outlettype" : [ "jit_matrix", "" ],
					"patching_rect" : [ 30.0, 735.0, 480.0, 270.0 ]
				}
			},
			{
				"box" : 				{
					"id" : "obj-43",
					"maxclass" : "comment",
					"numinlets" : 1,
					"numoutlets" : 0,
					"patching_rect" : [ 550.0, 735.0, 400.0, 200.0 ],
					"text" : "SIGNAL FLOW EXPLANATION:\n\n1. Video Input (jit.movie)\n   ↓\n2. sr.noise - Adds procedural noise (gaussian/row/column)\n   ↓\n3. sr.saturation - Boosts color intensity via HSV\n   ↓\n4. sr.chromatic - Shifts RGB channels for color fringing\n   ↓\n5. sr.corduroy - Adds scanner-like banding artifacts\n   ↓\n6. Output (jit.pwindow)\n\nEach effect receives the processed output from the previous\neffect, creating a cumulative glitch aesthetic. All effects\nrun on GPU using jit.gl.pix for real-time performance.\n\nTip: Try adjusting multiple parameters simultaneously\nto create unique combinations!"
				}
			}
		],
		"lines" : [ 			{
				"patchline" : 				{
					"source" : [ "obj-4", 0 ],
					"destination" : [ "obj-5", 0 ]
				}
			},
			{
				"patchline" : 				{
					"source" : [ "obj-5", 0 ],
					"destination" : [ "obj-6", 0 ]
				}
			},
			{
				"patchline" : 				{
					"source" : [ "obj-7", 0 ],
					"destination" : [ "obj-6", 0 ]
				}
			},
			{
				"patchline" : 				{
					"source" : [ "obj-6", 0 ],
					"destination" : [ "obj-16", 0 ],
					"midpoints" : [ 39.5, 220.0, 39.5, 310.0 ]
				}
			},
			{
				"patchline" : 				{
					"source" : [ "obj-11", 0 ],
					"destination" : [ "obj-12", 0 ]
				}
			},
			{
				"patchline" : 				{
					"source" : [ "obj-12", 0 ],
					"destination" : [ "obj-16", 0 ]
				}
			},
			{
				"patchline" : 				{
					"source" : [ "obj-14", 0 ],
					"destination" : [ "obj-15", 0 ]
				}
			},
			{
				"patchline" : 				{
					"source" : [ "obj-15", 0 ],
					"destination" : [ "obj-16", 0 ]
				}
			},
			{
				"patchline" : 				{
					"source" : [ "obj-16", 0 ],
					"destination" : [ "obj-22", 0 ],
					"midpoints" : [ 39.5, 340.0, 39.5, 430.0 ]
				}
			},
			{
				"patchline" : 				{
					"source" : [ "obj-20", 0 ],
					"destination" : [ "obj-21", 0 ]
				}
			},
			{
				"patchline" : 				{
					"source" : [ "obj-21", 0 ],
					"destination" : [ "obj-22", 0 ]
				}
			},
			{
				"patchline" : 				{
					"source" : [ "obj-22", 0 ],
					"destination" : [ "obj-31", 0 ],
					"midpoints" : [ 39.5, 460.0, 39.5, 550.0 ]
				}
			},
			{
				"patchline" : 				{
					"source" : [ "obj-26", 0 ],
					"destination" : [ "obj-27", 0 ]
				}
			},
			{
				"patchline" : 				{
					"source" : [ "obj-27", 0 ],
					"destination" : [ "obj-31", 0 ]
				}
			},
			{
				"patchline" : 				{
					"source" : [ "obj-29", 0 ],
					"destination" : [ "obj-30", 0 ]
				}
			},
			{
				"patchline" : 				{
					"source" : [ "obj-30", 0 ],
					"destination" : [ "obj-31", 0 ]
				}
			},
			{
				"patchline" : 				{
					"source" : [ "obj-31", 0 ],
					"destination" : [ "obj-40", 0 ],
					"midpoints" : [ 39.5, 580.0, 39.5, 670.0 ]
				}
			},
			{
				"patchline" : 				{
					"source" : [ "obj-35", 0 ],
					"destination" : [ "obj-36", 0 ]
				}
			},
			{
				"patchline" : 				{
					"source" : [ "obj-36", 0 ],
					"destination" : [ "obj-40", 0 ]
				}
			},
			{
				"patchline" : 				{
					"source" : [ "obj-38", 0 ],
					"destination" : [ "obj-39", 0 ]
				}
			},
			{
				"patchline" : 				{
					"source" : [ "obj-39", 0 ],
					"destination" : [ "obj-40", 0 ]
				}
			},
			{
				"patchline" : 				{
					"source" : [ "obj-40", 0 ],
					"destination" : [ "obj-42", 0 ],
					"midpoints" : [ 39.5, 700.0, 39.5, 730.0 ]
				}
			}
		]
	}
}
